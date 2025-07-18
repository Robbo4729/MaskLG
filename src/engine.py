import math
import sys
from typing import Iterable, Optional

import torch
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils

def dispatch_clip_grad(parameters, clip_value, mode='norm'):
    if mode == 'norm':
        return clip_grad_norm_(parameters, clip_value)
    elif mode == 'value':
        return clip_grad_value_(parameters, clip_value)
    else:
        raise ValueError(f"Unknown clip mode: {mode}")

class NativeScalerGA:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False, do_step=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        if do_step:
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, accumulation_step=1, mask_training=False, mask_only=False):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    count = 0

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        count += 1
        do_step = (count % accumulation_step == 0)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        is_second_order = False

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if do_step:
            if mask_only:
                mask_params = [p for n, p in model.named_parameters() if 'mask_logits' in n]
                optimizer.zero_grad()
                loss_scaler(loss, optimizer, clip_grad=max_norm,
                            parameters=mask_params, create_graph=is_second_order, do_step=do_step)
            else:
                optimizer.zero_grad()
                loss_scaler(loss, optimizer, clip_grad=max_norm,
                            parameters=model.parameters(), create_graph=is_second_order, do_step=do_step)
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
    header = 'Test:'

    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}