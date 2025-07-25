"""
Misc functions adapted for Lightning Fabric.
"""

import builtins
import datetime
import io
import os
import time
from collections import defaultdict, deque

import torch
from lightning.fabric import Fabric


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self, fabric: Fabric):
        """Synchronize count and total across processes using Fabric"""
        if fabric.world_size == 1:
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cpu")
        t = fabric.all_reduce(t, reduce_op="sum")
        self.count = int(t[0].item())
        self.total = t[1].item()

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, fabric: Fabric, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.fabric = fabric  # Store Fabric instance for distributed operations

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """Synchronize all meters across processes"""
        for meter in self.meters.values():
            meter.synchronize_between_processes(self.fabric)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if self.fabric.device.type == "cuda":
            log_msg.append("max mem: {mem}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0 or i + 1 == len(iterable):
                self.fabric.print(
                    log_msg.format(
                        i,
                        len(iterable),
                        eta=str(
                            datetime.timedelta(
                                seconds=int((len(iterable) - i) * iter_time.avg)
                            )
                        ),
                        meters=self,
                        time=iter_time,
                        data=data_time,
                        mem=(
                            torch.cuda.max_memory_allocated() / MB
                            if self.fabric.device.type == "cuda"
                            else 0
                        ),
                    )
                )
            i += 1
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.fabric.print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def _load_checkpoint_for_ema(model_ema, checkpoint):
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def gene_tranfer(args, cp, num_layers, load_head=False):
    cp_new = dict()

    all_shared = ["pos_embed", "patch_embed.proj.weight", "patch_embed.proj.bias"]
    for k in all_shared:
        cp_new[k] = cp[k]

    if load_head:
        cp_new["head.weight"], cp_new["head.bias"] = cp["head.weight"], cp["head.bias"]

    cp_new["fc_norm.weight"], cp_new["fc_norm.bias"] = (
        cp["norm.weight"],
        cp["norm.bias"],
    )

    for i in range(num_layers):
        k = "blocks." + str(i) + "."

        cp_new[k + "attn.qkv.weight"] = (
            i / num_layers * cp["blocks.block.attn.qkv.instances.0.weight_ilayer"]
            + cp["blocks.block.attn.qkv.instances.0.weight_base"]
        )
        cp_new[k + "attn.qkv.bias"] = (
            i / num_layers * cp["blocks.block.attn.qkv.instances.0.bias_ilayer"]
            + cp["blocks.block.attn.qkv.instances.0.bias_base"]
        )

        cp_new[k + "attn.proj.weight"] = (
            i / num_layers * cp["blocks.block.attn.proj.instances.0.weight_ilayer"]
            + cp["blocks.block.attn.qkv.instances.0.weight_base"]
        )
        cp_new[k + "attn.proj.bias"] = (
            i / num_layers * cp["blocks.block.attn.proj.instances.0.bias_ilayer"]
            + cp["blocks.block.attn.qkv.instances.0.bias_base"]
        )

        cp_new[k + "mlp.fc1.weight"] = (
            i / num_layers * cp["blocks.block.mlp.fc1.instances.0.weight_ilayer"]
            + cp["blocks.block.mlp.fc1.instances.0.weight_base"]
        )
        cp_new[k + "mlp.fc1.bias"] = (
            i / num_layers * cp["blocks.block.mlp.fc1.instances.0.bias_ilayer"]
            + cp["blocks.block.mlp.fc1.instances.0.bias_base"]
        )

        cp_new[k + "mlp.fc2.weight"] = (
            i / num_layers * cp["blocks.block.mlp.fc2.instances.0.weight_ilayer"]
            + cp["blocks.block.mlp.fc2.instances.0.weight_base"]
        )
        cp_new[k + "mlp.fc2.bias"] = (
            i / num_layers * cp["blocks.block.mlp.fc2.instances.0.bias_ilayer"]
            + cp["blocks.block.mlp.fc2.instances.0.bias_base"]
        )

        cp_new[k + "norm1.bias"] = (
            i / num_layers * cp["blocks.block.norm1.instances.0.bias_ilayer"]
            + cp["blocks.block.norm1.instances.0.bias_base"]
        )
        cp_new[k + "norm1.weight"] = (
            i / num_layers * cp["blocks.block.norm1.instances.0.weight_ilayer"]
            + cp["blocks.block.norm1.instances.0.weight_base"]
        )

        cp_new[k + "norm2.bias"] = (
            i / num_layers * cp["blocks.block.norm2.instances.0.bias_ilayer"]
            + cp["blocks.block.norm2.instances.0.bias_base"]
        )
        cp_new[k + "norm2.weight"] = (
            i / num_layers * cp["blocks.block.norm2.instances.0.weight_ilayer"]
            + cp["blocks.block.norm2.instances.0.weight_base"]
        )

    return cp_new
