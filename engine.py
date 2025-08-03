import math
import sys
from typing import Iterable, Optional

import torch
from lightning.fabric import Fabric
from timm.utils import ModelEma, accuracy
from torch.nn.utils import clip_grad_norm_
from torchmetrics import Accuracy, SumMetric, MeanMetric
from tqdm import tqdm

import utils
from losses import DistillationLoss
from masked_functions import get_mask_dict_with_RelaxedBernouli


def mask_train_one_epoch(
    fabric: Fabric,
    model: torch.nn.Module,
    ancestor_model: torch.nn.Module,
    criterion,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    set_training_mode=True,
    accumulation_step=1,
    mask_training=False,
    mask_only=False,
    temperature=1.0,
    sparse_reg=0.0,
):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(fabric=fabric, delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 50
    count = 0

    # 初始化变量保存最后的sparse_state
    final_sparse_state = None

    # samples: 输入数据   targets: 标签
    for samples, targets in (
        pbar := tqdm(
            metric_logger.log_every(data_loader, print_freq, header),
            dynamic_ncols=True,
            disable=not fabric.is_global_zero,
        )
    ):
        count += 1
        do_step = count % accumulation_step == 0

        # 使用Fabric的自动设备移动
        samples, targets = fabric.to_device((samples, targets))

        if targets.dtype == torch.long:
            targets = targets.float()

        # --- mask training logic ---
        mask_dict = {}
        if mask_training or mask_only:
            # 用RelaxedBernouli处理所有参数
            mask_dict = get_mask_dict_with_RelaxedBernouli(
                model.mask_logits, temperature=temperature, hard=False
            )

            # 应用mask到参数
            sparse_state = {}
            for name, param in ancestor_model.state_dict().items():
                if name in mask_dict:
                    sparse_state[name] = param * mask_dict[name]

            # 然后加载到 model
            # model.load_state_dict(sparse_state, strict=True)
            # outputs = model(samples)   不能直接拿model得到outputs,因为load_state_dict加载参数的方法会导致计算图断开，i.e., grad.fn不存在

        if fabric.is_global_zero:
            tqdm.write(
                f"Processing batch {count}, mask training: {mask_training}, mask only: {mask_only}"
            )

        with fabric.autocast():
            outputs = torch.func.functional_call(model, sparse_state, args=samples)
            # 判断criterion是否为DistillationLoss，并传入samples
            if isinstance(criterion, DistillationLoss):
                loss = criterion(samples, outputs, targets)
            else:
                loss = criterion(outputs, targets)
            acc = (torch.softmax(outputs, dim=-1).argmax(-1) == targets).mean()

        # 计算稀疏权重正则化项
        if sparse_reg > 0 and mask_training:
            reg_loss = 0.0
            for name, param in sparse_state.items():
                if name in mask_dict:
                    reg_loss += torch.sum(param**2)  # L2正则化
            loss = loss - sparse_reg * reg_loss

        loss_value = loss.item()

        if fabric.is_global_zero:
            pbar.set_postfix(loss=loss_value)

        if not math.isfinite(loss_value):
            fabric.print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if do_step:
            optimizer.zero_grad()

        fabric.backward(loss)

        if max_norm > 0:
            clip_grad_norm_(model.parameters(), max_norm)

        if do_step:
            optimizer.step()

        if model_ema is not None and do_step:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # 记录每个batch的loss
        fabric.log_dict(
            {
                "train/loss": loss_value,
                "train/acc": acc,
            },
            step=epoch * len(data_loader) + count,
        )

        # 保存最后一个batch的sparse_state
        if count == len(data_loader):
            final_mask_dict = get_mask_dict_with_RelaxedBernouli(
                model.mask_logits, temperature=temperature, hard=False
            )

            # 应用mask到参数
            final_sparse_state = {}
            for name, param in ancestor_model.state_dict().items():
                if name in final_mask_dict:
                    final_sparse_state[name] = param * final_mask_dict[name]

    metric_logger.synchronize_between_processes()
    if fabric.is_global_zero:
        fabric.print("Averaged stats:", metric_logger)

    return {
        "metrics": {k: meter.global_avg for k, meter in metric_logger.meters.items()},
        "final_sparse_state": final_sparse_state,
    }


@torch.no_grad()
def evaluate(fabric: Fabric, data_loader, model):
    # 只在主进程显示进度条
    disable_progress = not fabric.is_global_zero
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(fabric=fabric, delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images, target = fabric.to_device((images, target))

        with fabric.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()

    if fabric.is_global_zero:
        fabric.print(
            "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
                top1=metric_logger.acc1,
                top5=metric_logger.acc5,
                losses=metric_logger.loss,
            )
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
