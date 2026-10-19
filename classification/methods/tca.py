# Topological Consistency Adaptation (CVPR 2025).
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from augmentations.transforms_cotta import get_tta_transforms
from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY


@ADAPTATION_REGISTRY.register()
class TCA(TTAMethod):
    """Mean teacher adaptation with a persistent class-centroid graph."""

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        options = getattr(cfg, "TCA", None)

        def option(name, default):
            return getattr(options, name, default) if options is not None else default

        self.lambda_align = float(option("LAMBDA_ALIGN", 0.025))
        self.lambda_topology = float(option("LAMBDA_TOPOLOGY", 0.15))
        self.alpha = float(option("CENTROID_MOMENTUM", 0.9))
        self.beta = float(option("WEIGHT_MOMENTUM", 0.9))
        self.temperature = float(option("TEMPERATURE", 2.0))
        self.alignment_sign = float(option("ALIGNMENT_SIGN", -1.0))
        self.teacher_momentum = float(cfg.M_TEACHER.MOMENTUM)
        if not all(0 <= m < 1 for m in (self.alpha, self.beta, self.teacher_momentum)):
            raise ValueError("TCA EMA momenta must be in [0, 1).")
        if self.temperature <= 0:
            raise ValueError("TCA TEMPERATURE must be positive.")

        # Capture the input of the classifier without splitting/mutating the
        # backbone or copying a classifier outside the optimizer.
        candidates = [name for name, module in self.model.named_modules()
                      if isinstance(module, nn.Linear) and module.out_features == num_classes]
        if not candidates:
            raise ValueError("TCA requires a linear classification head with num_classes outputs.")
        self.classifier_name = candidates[-1]
        self.model_ema = self.copy_model(self.model)
        self.model_ema.requires_grad_(False)
        self.models = [self.model, self.model_ema]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()
        self.transform = get_tta_transforms(self.img_size)

        parameter = next(self.model.parameters())
        self.register_buffer("centroids", torch.empty(num_classes, 0, device=parameter.device))
        self.register_buffer("seen_classes", torch.zeros(num_classes, dtype=torch.bool, device=parameter.device))
        # EMA of node frequencies is algebraically equivalent to maintaining
        # every edge's EMA, since mu_ij = (mu_i + mu_j) / 2.
        self.register_buffer("class_weights", torch.zeros(num_classes, device=parameter.device))
        self.register_buffer("topology_steps", torch.zeros((), dtype=torch.long, device=parameter.device))

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(True)
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
            elif isinstance(module, nn.BatchNorm1d):
                module.train()

    def collect_params(self):
        # named_parameters() deduplicates shared parameters.
        pairs = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        return [p for _, p in pairs], [name for name, _ in pairs]

    def _student_forward(self, images):
        captured = []
        classifier = dict(self.model.named_modules())[self.classifier_name]

        def capture(module, args):
            captured.append(args[0])

        handle = classifier.register_forward_pre_hook(capture)
        try:
            logits = self.model(images)
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].ndim != 2:
            raise ValueError("TCA expects one [batch, feature_dim] classifier input per forward.")
        return logits, F.normalize(captured[0].float(), dim=1)

    def topology_loss(self, features, labels):
        """Eqs. (2)-(10); history is detached, current centroids retain gradients."""
        zero = features.sum() * 0.0
        counts = torch.bincount(labels, minlength=self.num_classes)
        present = counts > 0
        sums = features.new_zeros(self.num_classes, features.shape[1])
        sums.index_add_(0, labels, features)
        current = sums / counts.clamp_min(1).unsqueeze(1)

        if self.centroids.shape[1] == 0:
            self.centroids = torch.zeros_like(current)
        if self.centroids.shape != current.shape:
            raise ValueError("TCA feature dimension changed during adaptation.")
        history = self.centroids.detach().clone()
        smoothed = self.alpha * history + (1.0 - self.alpha) * current
        # Initialize unseen classes with their first actual centroid; absent
        # classes keep history, rather than introducing zero-valued graph nodes.
        updated = torch.where(self.seen_classes[:, None], smoothed, current)
        graph = torch.where(present[:, None], updated, history)
        active = self.seen_classes | present

        frequency = counts.to(features.dtype).sqrt()
        frequency = frequency / frequency.sum().clamp_min(1e-12)
        weights = self.beta * self.class_weights.detach().clone() + (1 - self.beta) * frequency
        first_batch = self.topology_steps.item() == 0
        if first_batch:
            weights = frequency

        nodes = graph[active]
        inter = zero
        if nodes.shape[0] > 1:
            pairs = torch.triu_indices(nodes.shape[0], nodes.shape[0], offset=1, device=features.device)
            distances_sq = (nodes[pairs[0]] - nodes[pairs[1]]).square().sum(1)
            potentials = -self.temperature * distances_sq
            if first_batch:  # Algorithm 1: use unweighted Eq. (4) first.
                inter = torch.logsumexp(potentials, 0) - math.log(potentials.numel())
            else:
                node_weights = weights[active].detach()
                edge_weights = (node_weights[pairs[0]] + node_weights[pairs[1]]) / 2
                valid = edge_weights > 0
                if valid.any():
                    inter = (torch.logsumexp(potentials[valid] + edge_weights[valid].log(), 0)
                             - edge_weights[valid].sum().log())

        # Eq. (5): average distances within a class BEFORE the exponential,
        # then average the potentials across classes with at least two samples.
        # Singleton classes contain no distinct pairs and do not contribute.
        within = []
        for class_id in present.nonzero(as_tuple=True)[0]:
            group = features[labels == class_id]
            n = group.shape[0]
            if n > 1:
                # Equivalent to the mean over all unordered pairs, O(n*d)
                # rather than allocating a pairwise feature-distance tensor.
                mean_distance = (2 * (n * group.square().sum() - group.sum(0).square().sum())
                                 / (n * (n - 1)))
                within.append(-self.temperature * mean_distance.clamp_min(0))
        intra = zero
        if within:
            intra = torch.logsumexp(torch.stack(within), 0) - math.log(len(within))

        with torch.no_grad():
            self.centroids.copy_(graph.detach())
            self.seen_classes.copy_(active)
            self.class_weights.copy_(weights.detach())
            self.topology_steps.add_(1)
        return inter + intra

    def loss_calculation(self, x):
        images = x[0]
        # Teacher predictions are generated before the update, with no teacher
        # gradients. Student augmented predictions supply Eq. (1) labels.
        with torch.no_grad():
            teacher_logits = self.model_ema(images)
        logits, features = self._student_forward(images)
        augmented_logits, augmented_features = self._student_forward(self.transform(images))
        teacher_logits = teacher_logits.float()
        logits = logits.float()
        sce = -(teacher_logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
        sce = sce - (logits.softmax(1) * teacher_logits.log_softmax(1)).sum(1).mean()
        alignment = self.alignment_sign * (features - augmented_features).square().sum(1).mean()
        topology = self.topology_loss(augmented_features, augmented_logits.detach().argmax(1))
        loss = sce + self.lambda_align * alignment + self.lambda_topology * topology
        return teacher_logits.detach(), loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        self.optimizer.zero_grad(set_to_none=True)
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs, loss = self.loss_calculation(x)
            loss.backward()
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for teacher, student in zip(self.model_ema.parameters(), self.model.parameters()):
                teacher.mul_(self.teacher_momentum).add_(student, alpha=1 - self.teacher_momentum)
            for teacher, student in zip(self.model_ema.buffers(), self.model.buffers()):
                if teacher.is_floating_point():
                    teacher.mul_(self.teacher_momentum).add_(student, alpha=1 - self.teacher_momentum)
                else:
                    teacher.copy_(student)
        return outputs

    @torch.no_grad()
    def forward_sliding_window(self, x):
        return self.model_ema(x[0])

    def reset(self):
        super().reset()
        self.optimizer.zero_grad(set_to_none=True)
        self.centroids.zero_()
        self.seen_classes.zero_()
        self.class_weights.zero_()
        self.topology_steps.zero_()
        self.input_buffer = None
        self.pointer.zero_()
        self.performed_updates = 0
