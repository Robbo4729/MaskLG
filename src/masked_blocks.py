import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedBlock(nn.Module):
    def __init__(self, block, temperature=1.0, threshold=0.5):
        super().__init__()
        self.block = block
        self.temperature = temperature
        self.threshold = threshold
        self.mask_logits = nn.ParameterDict()
        for name, param in self.block.named_parameters():
            if param.requires_grad:
                safe_name = name.replace('.', '_')
                # 更分散的初始化
                self.mask_logits[safe_name] = nn.Parameter(
                    torch.empty_like(param).uniform_(-2, 2), requires_grad=True
                )

    def forward(self, x):
        original_params = {}
        for name, param in self.block.named_parameters():
            safe_name = name.replace('.', '_')
            if safe_name in self.mask_logits:
                logits = self.mask_logits[safe_name]
                mask = F.gumbel_softmax(
                    torch.stack([logits, torch.zeros_like(logits)], dim=-1),
                    tau=self.temperature, hard=False
                )[..., 0]
                original_params[name] = param.data.clone()
                param.data = param.data * mask
        out = self.block(x)
        for name, param in self.block.named_parameters():
            if name in original_params:
                param.data = original_params[name]
        return out

    def get_masked_parameters(self):
        masked_params = {}
        for name, param in self.block.named_parameters():
            safe_name = name.replace('.', '_')
            if safe_name in self.mask_logits:
                mask = (torch.sigmoid(self.mask_logits[safe_name]) > self.threshold).float()
                masked_params[name] = param.data * mask
            else:
                masked_params[name] = param.data.clone()
        return masked_params

    def update_temperature(self, new_temp):
        self.temperature = new_temp