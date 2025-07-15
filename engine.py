import math
import sys
from typing import Iterable, Optional

import torch
from timm.data import Mixup
from timm.utils import ModelEma, accuracy
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

import utils
from losses import DistillationLoss
from masked_functions import get_mask_dict_with_combined_gumbel



def mask_train_one_epoch(
    model: torch.nn.Module,
    ancestor_model: torch.nn.Module,
    criterion,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    set_training_mode=True,
    accumulation_step=1,
    mask_training=False,
    mask_only=False,
    temperature=1.0,
    sparse_reg=0.0
):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 50
    count = 0

    # samples: 输入数据   targets: 标签
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        count += 1
        do_step = count % accumulation_step == 0
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # 确保 targets 为浮点类型
        if targets.dtype == torch.long:
            targets = targets.float()

        is_second_order = False

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # --- mask training logic ---
        mask_dict = {}
        if mask_training or mask_only:
            # 处理所有参数
            mask_dict = get_mask_dict_with_combined_gumbel(
                model.mask_logits,
                temperature=temperature,
                hard=False
            )

            # 应用mask到参数
            sparse_state = ancestor_model.state_dict()
            for name, mask in mask_dict.items():
                sparse_state[name] = sparse_state[name] * mask

            # 然后加载到 model
            model.load_state_dict(sparse_state, strict=True)

        print(
            f"Processing batch {count}, mask training: {mask_training}, mask only: {mask_only}"
        )

        outputs = model(samples)
        # 判断criterion是否为DistillationLoss，并传入samples
        if isinstance(criterion, DistillationLoss):
            loss = criterion(samples, outputs, targets)
        else:
            loss = criterion(outputs, targets)

        # 计算稀疏权重正则化项
        if sparse_reg > 0 and mask_training:
            reg_loss = 0.0
            for name, param in model.named_parameters():
                if name in mask_dict:
                    masked_weight = param * mask_dict[name]
                    reg_loss -= torch.sum(masked_weight ** 2)  # 负L2正则化
            loss += sparse_reg * reg_loss  # 加到总损失中

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if do_step:
            optimizer.zero_grad()  # 这里optimizer是特定block的优化器

        loss.backward()
        if max_norm > 0:
            clip_grad_norm_(model.parameters(), max_norm)
        
        if do_step:
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        if model_ema is not None and do_step:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
