import sys
import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
from copy import deepcopy
from methods.base import TTAMethod
from augmentations.transforms_cotta import get_tta_transforms
from utils.registry import ADAPTATION_REGISTRY
from utils.misc import ema_update_model
import os
from models.model import split_up_model
from utils.losses import SymmetricCrossEntropy

class EnergyModel(nn.Module):
    def __init__(self, model):
        super(EnergyModel, self).__init__()
        self.f = model

    def classify(self, x):
        penult_z = self.f(x)
        return penult_z

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1), logits
        else:
            return torch.gather(logits, 1, y[:, None]), logits


##############################################################################################################
@ADAPTATION_REGISTRY.register()
class CoTTA(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.mt = cfg.M_TEACHER.MOMENTUM
        self.rst = cfg.COTTA.RST
        self.ap = cfg.COTTA.AP
        self.n_augmentations = cfg.TEST.N_AUGMENTATIONS

        # Setup EMA and anchor/source model
        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        self.model_anchor = self.copy_model(self.model)  # Anchor model keeps original BN layers
        for param in self.model_anchor.parameters():
            param.detach_()

        self.models = [self.model, self.model_ema, self.model_anchor]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

        self.softmax_entropy = softmax_entropy_cifar if "cifar" in self.dataset_name else softmax_entropy_imagenet
        self.transform = get_tta_transforms(self.img_size)
        self.mytransform = get_tta_transforms_w()

        self.symmetric_cross_entropy = SymmetricCrossEntropy()
        self.m_teacher_momentum = cfg.M_TEACHER.MOMENTUM

        #####################################################################################################################

        # split up the model
        # cifar10c:
        arch_name = "Standard"
        dataset_name = "cifar10_c"

        # cifar100c:
        # arch_name = "Hendrycks2020AugMix_ResNeXt"
        # dataset_name = "cifar100_c"

        self.feature_extractor, self.classifier = split_up_model(self.model, arch_name, dataset_name)

        # Initialize smoothed class weights in __init__ method
        self.smoothed_weights_matrix = torch.ones(num_classes, num_classes).to(self.device) / num_classes
        self.centers = torch.zeros((self.num_classes, torch.ones(bs, Feature.shape)), device = features_test.device)
    ##############################################################################################################

    def loss_calculation(self, x):

        imgs_test = x[0]

        #######################################3333#######################################3333

        # # Create the prediction of the anchor (source) model
        anchor_prob = torch.nn.functional.softmax(self.model_anchor(imgs_test), dim=1).max(1)[0]

        # Augmentation-averaged Prediction
        ema_outputs = []
        if anchor_prob.mean(0) < self.ap:
            for _ in range(32):
                outputs_ = self.model_ema(self.transform(imgs_test)).detach()
                ema_outputs.append(outputs_)

            # Threshold choice discussed in supplementary
            outputs_ema = torch.stack(ema_outputs).mean(0)
        else:
            # Create the prediction of the teacher model
            outputs_ema = self.model_ema(imgs_test)

        features_test = self.feature_extractor(imgs_test)
        outputs_test = self.classifier(features_test)

        features_aug_test = self.feature_extractor(self.transform(imgs_test))
        outputs_aug_test = self.classifier(features_aug_test)
        # Calculate the number of samples for each class based on logits
        x_weak_features = self.feature_extractor(self.transform(imgs_test))

        with torch.no_grad():
            class_counts = torch.zeros(self.num_classes, device=outputs_test.device)
            predicted_classes = outputs_aug_test.argmax(dim=1)

            centers = torch.zeros((self.num_classes, features_test.size(1)), device=features_test.device)
            for i in range(self.num_classes):
                class_mask = (predicted_classes == i)
                class_counts[i] = class_mask.sum()
                if class_mask.sum() > 0:
                    # Calculate the centers
                    centers[i] = features_aug_test[class_mask].mean(dim=0)
                    self.centers= alpha * centers + (1-alpha) * self.centers
            epsilon = 1e-6  # Small constant to avoid division by zero
            class_weights = torch.sqrt(class_counts + epsilon)
            class_weights /= class_weights.sum()  # Normalize weights to sum to 1

            weights_matrix = (class_weights.unsqueeze(0) + class_weights.unsqueeze(1)) / 2
            # Smooth the class weights using exponential moving average (EMA)
            beta = 0.2 # [0,1.0]
            self.smoothed_weights_matrix = beta * weights_matrix + (1 - beta) * self.smoothed_weights_matrix

        #######################################3333#######################################3333
        loss_unif = self.calculate_uniformity_loss(centers, self.smoothed_weights_matrix)
        loss_compact = self.calculate_compactness_loss(features_aug_test, predicted_classes, centers, class_counts)

        loss_align_unif = 0.025 * (loss_unif + loss_compact) + 0.15 * lalign(x_weak_features, features_aug_test)
        loss_self_training = (0.5 * self.symmetric_cross_entropy(outputs_test, outputs_ema) + 0.5 * self.symmetric_cross_entropy(outputs_aug_test, outputs_ema)).mean(0)
        loss = loss_self_training +  loss_align_unif

        return 0.3 * outputs_test + 0.7 * outputs_ema, loss

    def calculate_uniformity_loss(self, centers, smoothed_weights_matrix, t=2.0):
        distance_matrix = torch.cdist(centers, centers, p=2).pow(2)
        upper_triangular_indices = torch.triu_indices(distance_matrix.size(0), distance_matrix.size(1), offset=1)
        distances = distance_matrix[upper_triangular_indices[0], upper_triangular_indices[1]]
        weights = smoothed_weights_matrix[upper_triangular_indices[0], upper_triangular_indices[1]]
        weighted_exp_dist = weights * distances.mul(-t).exp()
        uniformity_loss = weighted_exp_dist.mean().log()
        return uniformity_loss

    def calculate_compactness_loss(self, features, labels, centers, class_counts, t=2.0):
        # Calculate compactness loss using traditional uniformity loss form within each class
        compact_loss = 0.0
        for i in range(self.num_classes):
            class_mask = (labels == i)
            class_features = features[class_mask]
            sq_pdist = torch.pdist(class_features, p=2).pow(2)
            compact_loss += sq_pdist.mul(-t).exp().mean().log()
        return compact_loss / self.num_classes

    ########################################3333#######################################3333#############################3333##########
    @torch.enable_grad()
    def forward_and_adapt(self, x):

        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs_ema, loss = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs_ema, loss = self.loss_calculation(x)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

        self.model_ema = ema_update_model(
            model_to_update=self.model_ema,
            model_to_merge=self.model,
            momentum=self.m_teacher_momentum,
            device=self.device,
            update_all=True
        )
        return outputs_ema

    def configure_model(self):
        """Configure model."""
        # self.model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what we update
        self.model.requires_grad_(False)
        # enable all trainable
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()  # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            else:
                m.requires_grad_(True)

def collect_params(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


@torch.jit.script
def entropy(logits) -> torch.Tensor:
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_entropy_cifar(x, x_ema) -> torch.Tensor:
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


@torch.no_grad()
def update_model_probs(x_ema, x, momentum=0.9):
    return momentum * x_ema + (1 - momentum) * x


@torch.jit.script
def softmax_entropy_imagenet(x, x_ema) -> torch.Tensor:
    return -0.5 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - 0.5 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)





