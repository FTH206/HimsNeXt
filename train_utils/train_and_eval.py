import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR, OneCycleLR
import torch.nn.functional as F



def focal_loss(inputs, targets, alpha=0.6, gamma=2, reduction="mean"):

    ce_loss = nn.functional.cross_entropy(
        inputs, targets, reduction='none', ignore_index=255
    )


    pt = torch.exp(-ce_loss)

    focal_weight = (1 - pt) ** gamma

    if alpha is not None:

        alpha_weight = torch.ones_like(targets, device=targets.device)
        alpha_weight[targets == 1] = alpha
        alpha_weight[targets == 0] = 1 - alpha
        focal_weight = alpha_weight * focal_weight

    loss = focal_weight * ce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def criterion(inputs, target, num_classes: int = 2, dice: bool = True, ignore_index: int = -100,
              focal_alpha=0.25, focal_gamma=2):
    losses = {}

    if isinstance(inputs, list):
        n = len(inputs)
        base = 0.7
        weights = [base ** i for i in range(n)]
        s = sum(weights)
        weights = [w / s for w in weights]
        total_loss = 0
        for i, out in enumerate(inputs):

            if i > 0 and out.shape[2:] != target.shape[1:]:
                out = F.interpolate(out, size=target.shape[1:], mode='bilinear', align_corners=False)
            loss = focal_loss(out, target, alpha=focal_alpha, gamma=focal_gamma)
            if dice is True:
                dice_target = build_target(target, num_classes, ignore_index)
                loss += dice_loss(out, dice_target, multiclass=True, ignore_index=ignore_index)
            total_loss += weights[i] * loss
        return total_loss

    if isinstance(inputs, dict):
        for name, x in inputs.items():

            loss = focal_loss(x, target, alpha=focal_alpha, gamma=focal_gamma)

            if dice is True:
                dice_target = build_target(target, num_classes, ignore_index)
                loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
            losses[name] = loss
    else:

        loss = focal_loss(inputs, target, alpha=focal_alpha, gamma=focal_gamma)
        if dice is True:
            dice_target = build_target(target, num_classes, ignore_index)
            loss += dice_loss(inputs, dice_target, multiclass=True, ignore_index=ignore_index)
        losses['out'] = loss

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None, criterion_fn=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)

            if criterion_fn is not None:
                loss = criterion_fn(output, target)
            else:
                loss = criterion(output, target, num_classes=num_classes, ignore_index=255)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)

            if isinstance(output, list):
                output = output[0]
            elif isinstance(output, dict):
                output = output['out']

            elif isinstance(output, torch.Tensor):
                pass

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confmat, dice.value.item()


def create_lr_scheduler(optimizer, num_step: int, epochs: int,
                        warmup=True,
                        warmup_epochs=40,
                        T0=70, T_mult=2, eta_min=4e-5):
    if warmup:

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs * num_step
        )
        cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T0 * num_step,
            T_mult=T_mult,
            eta_min=eta_min
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs * num_step]
        )
    else:

        return OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            total_steps=epochs * num_step,
            pct_start=0.3,
            div_factor=10.0,
            final_div_factor=50.0,
            anneal_strategy='cos',
        )
