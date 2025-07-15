# filepath: /Users/xianglinluo/MaskLearngene/MaskLG/src/utils.py
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
from collections import defaultdict, deque
import torch
import torch.distributed as dist
import time
import datetime
import io
import os
import builtins


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

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

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
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr].avg
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {mem}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0 or i + 1 == len(iterable):
                print(log_msg.format(
                    i, len(iterable),
                    eta=str(datetime.timedelta(seconds=int((len(iterable) - i) * iter_time.avg))),
                    meters=self,
                    time=iter_time,
                    data=data_time,
                    mem=torch.cuda.max_memory_allocated() / MB if torch.cuda.is_available() else 0,
                ))
            i += 1
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def gene_tranfer(args, cp, num_layers, load_head=False):
    cp_new = dict()
    
    all_shared = ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias']
    for k in all_shared:
        cp_new[k] = cp[k]
    
    if load_head:
        cp_new['head.weight'], cp_new['head.bias'] = cp['head.weight'], cp['head.bias']
    
    cp_new['fc_norm.weight'], cp_new['fc_norm.bias'] = cp['norm.weight'], cp['norm.bias']

    for i in range(num_layers):
        k = 'blocks.'+str(i)+'.'

        cp_new[k+'attn.qkv.weight'] = \
            i/num_layers * cp['blocks.block.attn.qkv.instances.0.weight_ilayer'] + cp['blocks.block.attn.qkv.instances.0.weight_base']
        cp_new[k+'attn.qkv.bias'] = \
            i/num_layers * cp['blocks.block.attn.qkv.instances.0.bias_ilayer'] + cp['blocks.block.attn.qkv.instances.0.bias_base']

        cp_new[k+'attn.proj.weight'] = \
            i/num_layers * cp['blocks.block.attn.proj.instances.0.weight_ilayer'] + cp['blocks.block.attn.qkv.instances.0.weight_base']
        cp_new[k+'attn.proj.bias'] = \
            i/num_layers * cp['blocks.block.attn.proj.instances.0.bias_ilayer'] + cp['blocks.block.attn.qkv.instances.0.bias_base']
        
        cp_new[k+'mlp.fc1.weight'] = \
            i/num_layers * cp['blocks.block.mlp.fc1.instances.0.weight_ilayer'] + cp['blocks.block.mlp.fc1.instances.0.weight_base']
        cp_new[k+'mlp.fc1.bias'] = \
            i/num_layers * cp['blocks.block.mlp.fc1.instances.0.bias_ilayer'] + cp['blocks.block.mlp.fc1.instances.0.bias_base']

        cp_new[k+'mlp.fc2.weight'] = \
            i/num_layers * cp['blocks.block.mlp.fc2.instances.0.weight_ilayer'] + cp['blocks.block.mlp.fc2.instances.0.weight_base']
        cp_new[k+'mlp.fc2.bias'] = \
            i/num_layers * cp['blocks.block.mlp.fc2.instances.0.bias_ilayer'] + cp['blocks.block.mlp.fc2.instances.0.bias_base']
        
        cp_new[k+'norm1.bias'] = \
            i/num_layers * cp['blocks.block.norm1.instances.0.bias_ilayer'] + cp['blocks.block.norm1.instances.0.bias_base']
        cp_new[k+'norm1.weight'] = \
            i/num_layers * cp['blocks.block.norm1.instances.0.weight_ilayer'] + cp['blocks.block.norm1.instances.0.weight_base']

        cp_new[k+'norm2.bias'] = \
            i/num_layers * cp['blocks.block.norm2.instances.0.bias_ilayer'] + cp['blocks.block.norm2.instances.0.bias_base']
        cp_new[k+'norm2.weight'] = \
            i/num_layers * cp['blocks.block.norm2.instances.0.weight_ilayer'] + cp['blocks.block.norm2.instances.0.weight_base']
        
    return cp_new
