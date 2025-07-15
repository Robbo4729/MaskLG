import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def update_temperature(initial_temp, min_temp, epoch, epochs, decay_type):
    """更新温度"""
    if decay_type == "linear":
        current_temp = initial_temp - (initial_temp - min_temp) * (epoch / (epochs - 1))
    elif decay_type == "exp":
        alpha = epoch / epochs
        current_temp = initial_temp * (min_temp / initial_temp) ** alpha
    elif decay_type == "cosine":
        current_temp = min_temp + 0.5 * (initial_temp - min_temp) * (1 + math.cos(math.pi * epoch / epochs))
    else:
        current_temp = initial_temp

    # 确保返回值是浮点数且有效
    current_temp = float(current_temp) if current_temp is not None else min_temp
    return max(min(current_temp, initial_temp), min_temp)  # 确保在合理范围内

def get_mask_dict_with_combined_gumbel(mask_logits, temperature=1.0, hard=False):
    """
    使用合并后的logits应用Gumbel-Softmax采样,生成mask字典
    
    Args:
        model: 包含mask_logits的模型
        temperature: Gumbel-Softmax温度参数
        hard: 是否使用硬采样
        
    Returns:
        mask_dict: 生成的mask字典
    """
    # 收集所有logits的形状信息，用于后续重构
    logit_shapes = {}
    flattened_logits_list = []
    
    # 遍历所有logits，记录形状并展平
    for name, logits in mask_logits.items():
        logit_shapes[name] = logits.shape
        flattened_logits_list.append(logits.view(-1))
    
    # 合并所有展平后的logits
    combined_logits = torch.cat(flattened_logits_list, dim=0)
    
    # 应用Gumbel-Softmax采样
    combined_mask = torch.nn.functional.gumbel_softmax(
        combined_logits.unsqueeze(0),  # 添加batch维度
        tau=temperature,
        hard=hard
    ).squeeze(0)  # 移除batch维度
    
    # 将合并的mask分割并恢复原始形状
    mask_dict = {}
    start_idx = 0
    
    for name, shape in logit_shapes.items():
        num_elements = shape.numel()
        # 提取对应部分的mask
        flat_mask = combined_mask[start_idx:start_idx + num_elements]
        # 恢复原始形状
        mask_dict[name] = flat_mask.view(shape)
        start_idx += num_elements
    
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
    gumbel_mask = get_mask_dict_with_combined_gumbel(
        mask_logits,
        temperature=temperature,
        hard=False
    )
    
    hard_masks = {}
    for name, mask in gumbel_mask.items():
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
