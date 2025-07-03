import argparse
import torch
import torch.distributed as dist
import numpy as np
import time
import torch.backends.cudnn as cudnn
import os
import utils
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import get_state_dict, ModelEma
from MaskLG_vit import VisionTransformer
from datasets import build_dataset
from engine import train_one_epoch, evaluate, NativeScalerGA
from samplers import RASampler
from losses import DistillationLoss
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

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

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def get_masked_parameters(model, data_loader_train, num_blocks, criterion, epochs, lr, temperature, threshold, device, lambda_sparse=1e-3):
    # 为每个可训练参数新建mask_logits
    mask_logits = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.dim() > 1:  # 只对权重参数做mask
            mask_logits[name] = torch.nn.Parameter(torch.randn_like(param, device=device), requires_grad=True)
    optimizer = torch.optim.Adam(mask_logits.values(), lr=lr)

    for epoch in range(epochs):
        model.eval()
        for samples, targets in data_loader_train:
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            # Gumbel-Softmax采样mask
            mask_dict = {}
            for name, logits in mask_logits.items():
                mask = torch.nn.functional.gumbel_softmax(logits, tau=temperature, hard=False)
                mask_dict[name] = mask
            # 应用mask到参数
            sparse_state = model.state_dict()
            for name, mask in mask_dict.items():
                sparse_state[name] = sparse_state[name] * mask
            model.load_state_dict(sparse_state, strict=False)
            outputs = model(samples)
            loss = criterion(outputs, targets)
            sparse_loss = sum(mask.mean() for mask in mask_dict.values())
            total_loss = loss + lambda_sparse * sparse_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} mask training done.")

    # 生成二值mask并应用
    binary_mask = {}
    for name, logits in mask_logits.items():
        binary_mask[name] = (logits > threshold).float()
    sparse_state = model.state_dict()
    for name, mask in binary_mask.items():
        sparse_state[name] = sparse_state[name] * mask
    return sparse_state

def analyze_nonzero_parameters(model, print_details=False):
    total_nonzero = 0
    total_params = 0
    layer_stats = {}
    for name, param in model.named_parameters():
        numel = param.numel()
        nonzero = (param != 0).sum().item()
        layer_stats[name] = (nonzero, numel)
        total_nonzero += nonzero
        total_params += numel

    if print_details:
        print("\nNon-zero parameter distribution (sorted by block):")
        block_layers = []
        for name in layer_stats:
            if name.startswith("blocks."):
                parts = name.split('.')
                if len(parts) > 2 and parts[1].isdigit():
                    block_idx = int(parts[1])
                    subname = '.'.join(parts[2:])
                    block_layers.append((block_idx, subname, name))
        block_layers.sort()
        for block_idx, subname, name in block_layers:
            nonzero, numel = layer_stats[name]
            sparse = 100.0 * (1 - nonzero / numel)
            print(f"{name}: {nonzero}/{numel} ({sparse:.2f}% sparse)")
        for name in sorted(layer_stats.keys()):
            if not name.startswith("blocks."):
                nonzero, numel = layer_stats[name]
                sparse = 100.0 * (1 - nonzero / numel)
                print(f"{name}: {nonzero}/{numel} ({sparse:.2f}% sparse)")
    return total_nonzero, total_params, layer_stats

def main():
    parser = argparse.ArgumentParser('Learngene Extraction Example')
    parser.add_argument('--data_set', default='CIFAR10', type=str, help='Dataset name: CIFAR10 | CIFAR100 | IMNET')
    parser.add_argument('--data_path', default='./data', type=str, help='Path to dataset root')
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, help='Specify the model architecture to use (e.g., deit_base_patch16_224).')
    parser.add_argument('--num-blocks', default=12, type=int, help='Number of blocks in the model.')
    parser.add_argument('--epochs', default=10, type=int, help='Number of training epochs.')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate for the optimizer.')
    parser.add_argument('--temperature', default=1.0, type=float, help='Temperature parameter for Gumbel-Softmax.')
    parser.add_argument('--threshold', type=float, required=True, help='Threshold for binary mask generation (must be specified).')
    parser.add_argument('--device', default='cuda', type=str, help='Device to use for training (e.g., cuda or cpu).')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Start with pretrained version of specified network (if avail)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--print_details', action='store_true', help='Print per-layer nonzero parameter stats.')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training and evaluation.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for data loading.')
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--distributed', action='store_true', help='Use distributed training.')
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training.')
    parser.add_argument('--repeated_aug', action='store_true', help='Use repeated augmentation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--input_size', type=int, default=224, help='Input image size.')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.add_argument('--load-tar', action='store_true', help='Loading *.tar files for dataset')
    parser.set_defaults(repeated_aug=True)

    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=True, help='Enabling distributed evaluation')
    parser.add_argument('--ancestor-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of ancestor model to train (default: "regnety_160"')
    parser.add_argument('--ancestor-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    parser.add_argument('--finetune', default='', type=str, help='finetune from checkpoint')
    parser.set_defaults(train_mode=True)
    args = parser.parse_args()

    init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    seed = args.seed + utils.get_rank() if args.distributed else args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # dataset
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = DataLoader(
        dataset_train, 
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val, 
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters/(1000*1000))

    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() * args.accumulation_step / 512.0
        args.lr = linear_scaled_lr

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScalerGA()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    ancestor_model = None
    if args.distillation_type != 'none':
        print(f"Creating ancestor model: {args.ancestor_model}")
        # ancestor_pretrained is True when args.ancestor_path is empty
        ancestor_pretrained = not bool(args.ancestor_path)
        ancestor_model = create_model(
            args.ancestor_model,
            pretrained=ancestor_pretrained,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if not ancestor_pretrained:
            if args.ancestor_path.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.ancestor_path, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.ancestor_path, map_location='cpu')
            ancestor_model.load_state_dict(checkpoint['model'])
        ancestor_model.to(device)
        ancestor_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, ancestor_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    # Learngene Mask训练与提取
    learngene_params = get_masked_parameters(
        ancestor_model,
        data_loader_train,
        num_blocks=args.num_blocks,
        criterion=criterion,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        threshold=args.threshold,
        device=args.device
    )

    aux_model = VisionTransformer(depth=args.num_blocks, mask_training=False).to(args.device)
    aux_model.load_state_dict(learngene_params, strict=False)
    print(f"Auxiliary model parameter count after training: {sum(p.numel() for p in aux_model.parameters())}")

    total_nonzero, total_params, layer_stats = analyze_nonzero_parameters(aux_model, args.print_details)
    print(f"\nTotal parameters in auxiliary model: {total_params}")
    print(f"Non-zero parameters in auxiliary model: {total_nonzero} ({100.0 * total_nonzero / total_params:.2f}%)")
    print("Learngene successfully extracted and loaded into auxiliary model")

if __name__ == '__main__':
    main()