'''
Takes str argument and returns model 
'''

from monai.networks.nets import UNet, SwinUNETR, UNETR
from torch import nn

import torch



def model_loader(model_name:str, task:str='classification', n_classes:int=5, pretrained:bool=True):
    
    assert task in ['classification', 'segmentation'] , 'Only classification and segmentation tasks are supported!'
    print(f">>>>>>>>>>>  Model Name: {model_name} <<<<<<<<<<<<<")

    ############# 3D segmentation
    if model_name == 'unet':
        model = UNet(
            spatial_dims=3, 
            in_channels=1, 
            out_channels=n_classes, 
            channels=[64,128,512,128], 
            strides=[2,2,2,2], 
            kernel_size=3, 
            up_kernel_size=3, 
            num_res_units=4, 
            act='PRELU', 
            norm='INSTANCE', 
            dropout=0.0, 
            bias=True, 
            adn_ordering='NDA'
            )
     

    elif model_name == 'swin':
        model = SwinUNETR(
            img_size=96, 
            in_channels=1, 
            out_channels=3, 
            depths=(2, 2, 2, 2), 
            num_heads=(3, 6, 12, 24), 
            feature_size=24, 
            norm_name='instance', 
            drop_rate=0.0, 
            attn_drop_rate=0.0, 
            dropout_path_rate=0.0, 
            normalize=True, 
            use_checkpoint=False, 
            spatial_dims=3, 
            downsample='merging', 
            use_v2=False
            )
        

    elif model_name == 'unetr':
        model = UNETR(
            in_channels=1, 
            out_channels=3, 
            img_size=96, 
            feature_size=16, 
            hidden_size=768, 
            mlp_dim=3072, 
            num_heads=12, 
            pos_embed='conv', 
            proj_type='conv', 
            norm_name='instance', 
            conv_block=True, 
            res_block=True, 
            dropout_rate=0.0, 
            spatial_dims=3, 
            qkv_bias=False, 
            save_attn=False)

 
    return model