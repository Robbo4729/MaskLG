import torch
from timm.models.registry import register_model
from models import *



def get_repeated_schedule(depth):
    return {
        'norm1': [[depth], [True]], 
        'norm2': [[depth], [True]], 
        'attn_rpe': [[depth], [True]], 
        'attn_qkv': [[depth], [True]],  
        'attn_proj': [[depth], [True]], 
        'mlp_fc1': [[depth], [True]], 
        'mlp_fc2': [[depth], [True]],
    }



@register_model
def aux_deit_small_patch16_224(pretrained=False, **kwargs):
    return deit_small_patch16_224(pretrained=pretrained,
                                  use_cls_token=False,
                                  repeated_times_schedule=get_repeated_schedule(12),
                                  **kwargs)


@register_model
def aux_deit_base_patch16_224(pretrained=False, **kwargs):
    return deit_base_patch16_224(pretrained=pretrained,
                                 use_cls_token=False,
                                 repeated_times_schedule=get_repeated_schedule(12),
                                 **kwargs)



