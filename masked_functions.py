】import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist


def update_temperature(initial_temp, min_temp, epoch, epochs, decay_type):
    """更新温度"""
    if decay_type == "linear":
        current_temp = initial_temp - (initial_temp - min_temp) * (epoch / (epochs - 1))
    elif decay_type == "exp":
        alpha = epoch / epochs
        current_temp = initial_temp * (min_temp / initial_temp) ** alpha
    elif decay_type == "cosine":
        current_temp = min_temp + 0.5 * (initial_temp - min_temp) * (
            1 + math.cos(math.pi * epoch / epochs)
        )
    else:
        current_temp = initial_temp

    # 确保返回值是浮点数且有效
    current_temp = float(current_temp) if current_temp is not None else min_temp
    return max(min(current_temp, initial_temp), min_temp)  # 确保在合理范围内


def get_mask_dict_with_RelaxedBernouli(mask_logits, temperature=1.0, hard=False):
    """
    使用 Relaxed Bernoulli 分布生成可微的二进制掩码。
    
    参数:
        mask_logits (dict): 各参数对应的掩码 logits(未归一化的概率)。
        temperature (float): Gumbel-Softmax 温度参数，控制松弛程度。
        hard (bool): 是否返回硬掩码（二值化，但梯度仍可传播）。
    
    返回:
        dict: 参数名到对应掩码的字典, 掩码值为连续值(0~1)或硬二值(0或1)。
    """
    mask_dict = {}
    for name, logits in mask_logits.items():
        # 生成 Relaxed Bernoulli 分布的掩码
        relaxed_bernoulli = dist.RelaxedBernoulli(
            temperature=temperature,
            logits=logits
        )
        mask = relaxed_bernoulli.rsample()  # 可微采样
        
        # 硬二值化（Straight-Through Estimator）
        if hard:
            hard_mask = (mask > 0.5).float()
            mask = hard_mask + (mask - mask.detach())  # 保持梯度
        
        mask_dict[name] = mask
    
    return mask_dict


def get_binary_mask(mask_logits, threshold, temperature=0.1):
    """生成二值mask

    Args:
        mask_logits: 包含所有mask logits的字典
        threshold: 二值化阈值
        temperature: Gumbel-Softmax温度参数 (设置为低温度)

    Returns:
        binary_masks: 生成的二值mask字典
    """
    # 使用get_mask_dict_with_combined_gumbel处理所有logits
    Bernouli_mask = get_mask_dict_with_RelaxedBernouli(
        mask_logits, temperature=temperature, hard=False
    )

    hard_masks = {}
    for name, mask in Bernouli_mask.items():
        hard_masks[name] = (mask > threshold).float()

    return hard_masks


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
                parts = name.split(".")
                if len(parts) > 2 and parts[1].isdigit():
                    block_idx = int(parts[1])
                    subname = ".".join(parts[2:])
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
