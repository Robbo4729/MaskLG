import argparse
import datetime
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, get_state_dict
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler

import models
import utils
from datasets import build_dataset
from engine import evaluate, mask_train_one_epoch
from losses import DistillationLoss
from masked_functions import update_temperature, get_binary_mask, analyze_nonzero_parameters
from samplers import RASampler


def get_args_parser():
    parser = argparse.ArgumentParser("MaskLG training", add_help=False)
    #决定进行哪个阶段（Stage 1: Extracting learngene or Stage 2: Building descendent models）
    parser.add_argument(
        "--stage",
        default="stage1",
        type=str,
        choices=["stage1", "stage2"],
        help="Stage of training: stage1 for extracting learngene, stage2 for building descendant models.",
    )
    parser.add_argument(
        "--data-set",
        default="CIFAR10",
        type=str,
        help="Dataset name: CIFAR10 | CIFAR100 | IMNET",
    )
    parser.add_argument(
        "--data-path", default="./data", type=str, help="Path to dataset root"
    )
    parser.add_argument(
        "--output-dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument("--unscale-lr", action="store_true")
    parser.add_argument(
        "--model",
        default="",
        type=str,
        help="Specify the model architecture to use (e.g., deit_base_patch16_224).",
    )
    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop-path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )
    parser.add_argument(
        "--num-blocks", default=12, type=int, help="Number of blocks in the model."
    )
    parser.add_argument(
        "--epochs", default=10, type=int, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr", default=0.01, type=float, help="Learning rate for the optimizer."
    )
    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        help="Temperature parameter for Gumbel-Softmax.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Threshold for binary mask generation (must be specified).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Device to use for training (e.g., cuda or cpu).",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="Start with pretrained version of specified network (if avail)",
    )
    parser.add_argument(
        "--print_details",
        action="store_true",
        help="Print per-layer nonzero parameter stats.",
    )
    parser.add_argument(
        "--batch-size",
        default=128,
        type=int,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--num-workers", default=4, type=int, help="Number of workers for data loading."
    )
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument(
        "--distributed", action="store_true", help="Use distributed training."
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="url used to set up distributed training.",
    )
    parser.add_argument(
        "--repeated_aug", action="store_true", help="Use repeated augmentation."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--input_size", type=int, default=224, help="Input image size.")
    parser.add_argument(
        "--color-jitter",
        type=float,
        default=0.4,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)',
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--train-interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    parser.add_argument("--repeated-aug", action="store_true")
    parser.add_argument("--no-repeated-aug", action="store_false", dest="repeated_aug")
    parser.add_argument(
        "--load-tar", action="store_true", help="Loading *.tar files for dataset"
    )
    parser.set_defaults(repeated_aug=True)

    parser.add_argument("--train-mode", action="store_true")
    parser.add_argument(
        "--accumulation-step", type=int, default=1, help="gradient accumulation steps"
    )
    parser.add_argument("--no-train-mode", action="store_false", dest="train_mode")
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--eval-crop-ratio", default=0.875, type=float, help="Crop ratio for evaluation"
    )
    parser.add_argument(
        "--dist-eval",
        action="store_true",
        default=True,
        help="Enabling distributed evaluation",
    )
    parser.add_argument(
        "--ancestor-model",
        default="regnety_160",
        type=str,
        metavar="MODEL",
        help='Name of ancestor model to train (default: "regnety_160"',
    )
    parser.add_argument("--ancestor-path", type=str, default="")
    parser.add_argument(
        "--distillation-type",
        default="none",
        choices=["none", "soft", "hard"],
        type=str,
        help="",
    )
    parser.add_argument("--distillation-alpha", default=0.5, type=float, help="")
    parser.add_argument("--distillation-tau", default=1.0, type=float, help="")

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")

    parser.add_argument("--model-ema", action="store_true")
    parser.add_argument("--no-model-ema", action="store_false", dest="model_ema")
    parser.set_defaults(model_ema=True)
    parser.add_argument("--model-ema-decay", type=float, default=0.99996, help="")
    parser.add_argument(
        "--model-ema-force-cpu", action="store_true", default=False, help=""
    )
    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt-eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine"',
    )
    # * Mixup params
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.8,
        help="mixup alpha, mixup enabled if > 0. (default: 0.8)",
    )
    parser.add_argument(
        "--cutmix",
        type=float,
        default=1.0,
        help="cutmix alpha, cutmix enabled if > 0. (default: 1.0)",
    )
    parser.add_argument(
        "--cutmix-minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup-switch-prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup-mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )
    #温度衰减参数
    parser.add_argument(
        "--min-temperature",
        default=0.1,
        type=float,
        help="Minimum temperature for Gumbel-Softmax annealing.",
    )
    parser.add_argument(
        "--temperature-decay",
        default="linear",
        type=str,
        choices=["linear", "exp", "cosine"],
        help="Temperature decay schedule type.",
    )
    #稀疏正则化参数
    parser.add_argument(
        "--sparse-reg",
        type=float,
        default=1e-5,  # 默认值，可根据实验调整
        help="Sparse weight regularization strength (default: 1e-5)",
    )
    parser.set_defaults(train_mode=True)
    return parser


def main(args):

    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    if args.distillation_type != "none" and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    seed = args.seed + utils.get_rank() if args.distributed else args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # dataset
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed
        print(
            f"Creating distributed samplers for training and validation datasets with {args.data_set} dataset."
        )
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
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
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
        drop_last=False,
    )

    #定义损失函数
    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.0:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    #创建 ancestor model
    ancestor_model = None
    if args.distillation_type != "none":
        print(f"Creating ancestor model: {args.ancestor_model}")
        # ancestor_pretrained is True when args.ancestor_path is empty
        ancestor_pretrained = not bool(args.ancestor_path)
        ancestor_model = create_model(
            args.ancestor_model,
            pretrained=ancestor_pretrained,
            num_classes=args.nb_classes,
            #global_pool="avg",
        )
        if not ancestor_pretrained:
            if args.ancestor_path.startswith("https"):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.ancestor_path, map_location="cpu", check_hash=True
                )
            else:
                checkpoint = torch.load(args.ancestor_path, map_location="cpu")

            # 获取当前模型的状态字典
            state_dict = ancestor_model.state_dict()

            ancestor_model.load_state_dict(checkpoint["model"])
        ancestor_model.to(device)
        ancestor_model.eval()

    n_parameters = sum(
        p.numel() for p in ancestor_model.parameters() if p.requires_grad
    )
    print("number of params in ancestor_model:", n_parameters)

    #For stage 1: Construct auxiliary model
    if args.stage == "stage1":
        if ancestor_model is not None:
            #model为辅助模型
            model = create_model(
                args.ancestor_model,
                pretrained=False,  # 不加载预训练参数
                num_classes=args.nb_classes,
                drop_rate=0.0,    # 保持与ancestor_model相同的配置
                drop_path_rate=0.0,
            )
            model.load_state_dict(ancestor_model.state_dict(), strict=True)
            model.to(device)

    #For Stage 2: Construct descendant models
    if args.stage == "stage2":
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
        )

        if args.finetune:
            if args.finetune.startswith("https"):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.finetune, map_location="cpu", check_hash=True
                )
            else:
                checkpoint = torch.load(args.finetune, map_location="cpu")

            checkpoint_model = checkpoint
            state_dict = model.state_dict()
            for k in [
                "head.weight",
                "head.bias",
                "head_dist.weight",
                "head_dist.bias",
            ]:
                if (
                    k in checkpoint_model
                    and checkpoint_model[k].shape != state_dict[k].shape
                ):
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            pos_embed_checkpoint = checkpoint_model["pos_embed"]
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            new_size = int(num_patches**0.5)
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed

            model.load_state_dict(checkpoint_model, strict=False)

        model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if not args.unscale_lr:
        linear_scaled_lr = (
            args.lr
            * args.batch_size
            * utils.get_world_size()
            * args.accumulation_step
            / 512.0
        )
        args.lr = linear_scaled_lr

    if args.stage == "stage2":
        optimizer = create_optimizer(args, model_without_ddp)

        lr_scheduler, _ = create_scheduler(args, optimizer)

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion,
        ancestor_model,
        args.distillation_type,
        args.distillation_alpha,
        args.distillation_tau,
    )

    output_dir = Path(args.output_dir)
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    epochs=args.epochs
    lr=args.lr
    threshold=args.threshold
    initial_temp = args.temperature
    min_temp = args.min_temperature

    #进入mask训练阶段
    if args.stage == "stage1":

        # 为每个可训练参数新建mask_logits
        model_without_ddp.mask_logits = {}
        for name, param in model_without_ddp.named_parameters():
            if param.requires_grad: #and param.dim() > 1 
                model_without_ddp.mask_logits[name] = torch.nn.Parameter(torch.randn_like(param, device=device), requires_grad=True)
    
        # 创建单个优化器优化所有mask_logits
        mask_optimizer = torch.optim.Adam(model_without_ddp.mask_logits.values(), lr=args.lr)
        # 为mask_optimizer创建学习率调度器
        lr_scheduler, _ = create_scheduler(args, mask_optimizer)
            
        for epoch in range(epochs):
            # 更新温度
            current_temp = update_temperature(initial_temp, min_temp, epoch, epochs, args.temperature_decay)

            print(f"Epoch {epoch+1}/{epochs} with temperature: {current_temp:.4f}")
                
            test_stats = mask_train_one_epoch(
                model_without_ddp,
                ancestor_model,
                criterion,
                data_loader_train,
                mask_optimizer,
                device,
                epoch,
                max_norm=0,
                model_ema=None,
                mixup_fn=None,
                set_training_mode=args.train_mode,
                accumulation_step=1,
                mask_training=True,
                mask_only=True,
                temperature=current_temp,
                sparse_reg=args.sparse_reg # 稀疏正则化参数
            )

            lr_scheduler.step(epoch)
                
            # 评估当前block训练后的效果
            test_stats = evaluate(data_loader_val, model, device)
            print(f"Epoch {epoch+1}/{epochs} - Accuracy: {test_stats['acc1']:.1f}%")


        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Total training time {}".format(total_time_str))

        # Save checkpoint
        if args.output_dir and test_stats["acc1"] > max_accuracy:
            max_accuracy = test_stats["acc1"]
            checkpoint = {
                'model': model.state_dict(),
                'args': args,
                'mask_logits': model_without_ddp.mask_logits.state_dict(),
                'optimizer': mask_optimizer.state_dict(),
                'epoch': epoch,
            }
        
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint_best.pth')
            )

        # After training, generate binary mask for stage1
        hard_mask = get_binary_mask(model_without_ddp.mask_logits, args.threshold, temperature = 0.1)

        #Apply mask to get final parameters
        Learngene = model.state_dict()
        for name, mask in hard_mask.items():
            Learngene[name] = Learngene[name] * mask
        
        # Save final masked model
        if args.output_dir:
            torch.save({
                'model': Learngene,
                'mask': hard_mask,
            }, os.path.join(args.output_dir, 'masked_model.pth'))

    print(
        f"Auxiliary model parameter count after training: {sum(p.numel() for p in model.parameters())}"
    )

    total_nonzero, total_params, layer_stats = analyze_nonzero_parameters(
        model, args.print_details
    )
    print(f"\nTotal parameters in auxiliary model: {total_params}")
    print(
        f"Non-zero parameters in auxiliary model: {total_nonzero} ({100.0 * total_nonzero / total_params:.2f}%)"
    )
    print("Learngene successfully extracted and loaded into auxiliary model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MaskLG training", parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
