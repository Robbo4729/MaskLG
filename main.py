import argparse
import datetime
import os
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from lightning.fabric import Fabric
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, get_state_dict
from torch.distributed import init_process_group
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from lightning.fabric.loggers.tensorboard import TensorBoardLogger
import utils
from datasets import build_dataset
from engine import evaluate, mask_train_one_epoch
from losses import DistillationLoss
from masked_functions import (
    get_binary_mask,
    update_temperature,
)
from samplers import RASampler


def get_args_parser():
    parser = argparse.ArgumentParser("MaskLG training", add_help=False)
    # 决定进行哪个阶段（Stage 1: Extracting learngene or Stage 2: Building descendent models）
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

    # Learning rate schedule parameters
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine"',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--lr-noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr-noise-pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="learning rate noise limit percent (default: 0.67)",
    )
    parser.add_argument(
        "--lr-noise-std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="learning rate noise std-dev (default: 1.0)",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="warmup learning rate (default: 1e-5)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=30,
        metavar="N",
        help="epoch interval to decay LR",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=10,
        metavar="N",
        help="patience epochs for Plateau LR scheduler (default: 10",
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )

    # 温度衰减参数
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
    # 稀疏正则化参数
    parser.add_argument(
        "--sparse-reg",
        type=float,
        default=1e-6,  # 默认值，可根据实验调整
        help="Sparse weight regularization strength (default: 1e-6)",
    )
    parser.add_argument("--num_workers", default=0, type=int)
    parser.set_defaults(train_mode=True)
    return parser


def main(args):

    # 初始化 Lightning Fabric
    fabric = Fabric(
        accelerator="auto",
        devices="auto",
        strategy="ddp",
        loggers=TensorBoardLogger(root_dir="outputs"),
    )
    fabric.launch()

    fabric.print(args)

    device = fabric.device

    if args.distillation_type != "none" and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    if args.seed is not None:
        L.seed_everything(args.seed)

    # dataset
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,  # per-device batch size, global batch size = per-device batch size * number of gpu
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        shuffle=True,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False,
    )
    data_loader_train, data_loader_val = fabric.setup_dataloaders(
        data_loader_train, data_loader_val
    )

    # 创建 ancestor model
    ancestor_model = None
    if args.distillation_type != "none":
        fabric.print(f"Creating ancestor model: {args.ancestor_model}")
        # ancestor_pretrained is True when args.ancestor_path is empty
        ancestor_pretrained = not bool(args.ancestor_path)
        ancestor_model = create_model(
            args.ancestor_model,
            pretrained=ancestor_pretrained,
            num_classes=args.nb_classes,
            # global_pool="avg",
        )
        if not ancestor_pretrained:
            if args.ancestor_path.startswith("https"):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.ancestor_path, map_location="cpu", check_hash=True
                )
            else:
                checkpoint = fabric.load(args.ancestor_path)

            # 获取当前模型的状态字典
            state_dict = ancestor_model.state_dict()

            ancestor_model.load_state_dict(checkpoint["model"])
        ancestor_model = fabric.setup_module(ancestor_model)
        ancestor_model.eval()

    n_parameters = sum(
        p.numel() for p in ancestor_model.parameters() if p.requires_grad
    )
    fabric.print("number of params in ancestor_model:", n_parameters)

    # For stage 1: Construct auxiliary model
    if args.stage == "stage1":
        if ancestor_model is not None:
            # model为辅助模型
            model = create_model(
                args.ancestor_model,
                pretrained=False,  # 不加载预训练参数
                num_classes=args.nb_classes,
                drop_rate=0.0,  # 保持与ancestor_model相同的配置
                drop_path_rate=0.0,
            )
            model.load_state_dict(ancestor_model.state_dict(), strict=True)
            model = fabric.setup_module(model)

    # For Stage 2: Construct descendant models
    if args.stage == "stage2":
        # model为子代模型
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
                checkpoint = fabric.load(args.finetune)

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
                    fabric.print(f"Removing key {k} from pretrained checkpoint")
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
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed

            model.load_state_dict(checkpoint_model, strict=False)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )

    # Fabric will handle the DDP wrapping
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module

    if not args.unscale_lr:
        linear_scaled_lr = (
            args.lr
            * args.batch_size
            * fabric.world_size
            * args.accumulation_step
            / 512.0
        )
        args.lr = linear_scaled_lr

    if args.stage == "stage2":
        optimizer = create_optimizer(args, model)
        model, optimizer = fabric.setup(model, optimizer)
        lr_scheduler, _ = create_scheduler(args, optimizer)

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        torch.nn.CrossEntropyLoss(),  # base_criterion
        ancestor_model,
        args.distillation_type,
        args.distillation_alpha,
        args.distillation_tau,
    )

    if args.eval:
        test_stats = evaluate(fabric, data_loader_val, model, device)
        fabric.print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        return

    fabric.print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    epochs = args.epochs
    initial_temp = args.temperature
    min_temp = args.min_temperature

    # 评估祖先模型性能
    evaluate(fabric, data_loader_val, ancestor_model)

    # 获取原始模型
    model_without_prefix = model.module if hasattr(model, "module") else model

    # 进入mask训练阶段
    if args.stage == "stage1":

        # 为每个可训练参数新建mask_logits
        model_without_prefix.mask_logits = {}
        for name, param in model_without_prefix.named_parameters():
            if param.requires_grad:  # and param.dim() > 1
                model_without_prefix.mask_logits[name] = torch.nn.Parameter(
                    torch.zeros_like(param, device=device), requires_grad=True
                )

        # 创建单个优化器优化所有mask_logits
        mask_optimizer = torch.optim.Adam(
            model_without_prefix.mask_logits.values(), lr=args.lr
        )
        mask_optimizer = fabric.setup_optimizers(mask_optimizer)

        # 为mask_optimizer创建学习率调度器
        lr_scheduler, _ = create_scheduler(args, mask_optimizer)

        for epoch in range(epochs):
            # 更新温度
            current_temp = update_temperature(
                initial_temp, min_temp, epoch, epochs, args.temperature_decay
            )

            fabric.print(
                f"Epoch {epoch+1}/{epochs} with temperature: {current_temp:.4f}"
            )

            train_result = mask_train_one_epoch(
                fabric,
                model_without_prefix,
                ancestor_model,
                criterion,
                data_loader_train,
                mask_optimizer,
                epoch,
                max_norm=0,
                model_ema=None,
                set_training_mode=args.train_mode,
                accumulation_step=args.accumulation_step,
                mask_training=True,
                mask_only=True,
                temperature=current_temp,
                sparse_reg=args.sparse_reg,  # 稀疏正则化参数
            )

            lr_scheduler.step(epoch)

            # 在评估前加载最新的参数
            if train_result["final_sparse_state"] is not None:
                model.load_state_dict(train_result["final_sparse_state"], strict=True)

            # 评估训练后的效果
            test_stats = evaluate(fabric, data_loader_val, model)

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            fabric.print("Total training time {}".format(total_time_str))

            if fabric.is_global_zero:
                fabric.print(f"Epoch {epoch+1} - Accuracy: {test_stats['acc1']:.1f}%")

                # Save checkpoint
                if fabric.is_global_zero and test_stats["acc1"] > max_accuracy:
                    max_accuracy = test_stats["acc1"]
                    checkpoint = {
                        "model": model.state_dict(),
                        "args": args,
                        "mask_logits": model.mask_logits,
                        "optimizer": mask_optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    fabric.save(
                        os.path.join(args.output_dir, "checkpoint_best.pth"), checkpoint
                    )

        # After training, generate binary mask for stage1
        hard_mask = get_binary_mask(model.mask_logits, args.threshold, temperature=0.1)

        # Apply mask to get final parameters
        Learngene = model.state_dict()
        for name, mask in hard_mask.items():
            Learngene[name] = Learngene[name] * mask

        # Save final masked model
        if args.output_dir and fabric.is_global_zero:
            fabric.save(
                {
                    "model": Learngene,
                    "mask": hard_mask,
                },
                os.path.join(args.output_dir, "masked_model.pth"),
            )

        fabric.print("Learngene successfully extracted and loaded into auxiliary model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MaskLG training", parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
