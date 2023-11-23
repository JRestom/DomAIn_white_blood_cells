'''
Takes str argument and returns model 
'''
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from monai.networks.nets import UNet
from torch import nn
import open_clip
from utils.unet import UNet2D



class biomed_class(nn.Module):
                def __init__(self, n_classes):
                    super().__init__()
                    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                    # print(model)
                    self.encoder = model.visual
                    for _, param in self.encoder.named_parameters():
                        param.requires_grad = False
                    self.head = nn.Linear(512, n_classes)

                def forward(self, image):
                    out = self.encoder(image)
                    pred = self.head(out)

                    return pred



class biomed_seg(nn.Module):
                def __init__(self, n_classes):
                    super().__init__()
                    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                    # print(model)
                    self.encoder = model.visual
                    for _, param in self.encoder.named_parameters():
                        param.requires_grad = False
                    unet = UNet2D(
                        in_channels=1, 
                        out_channels=3
                                )
                    
                    decoder = unet.decoder_layers
                    print("*"*50)
                    print("BiomedCLIP Encoder ...\n")
                    print(self.encoder)
                    print("*"*50, '\n\n')
                    print("*"*50)
                    print("UNet2D Decoder ...\n")
                    print(decoder)
                    print("*"*50)
                    # for name, module in unet.named_modules():
                    #     print(name)

                # def forward(self, image):
                #     out = self.encoder(image)
                #     pred = self.head(out)

                #     return pred




def model_loader(model_name:str, task:str='classification', n_classes:int=5, pretrained:bool=True):
    
    assert task in ['classification', 'segmentation'] , 'Only classification and segmentation tasks are supported!'
    
    if task =='classification':
        if model_name == 'resnet':
            if pretrained:
                weights = ResNet50_Weights
            model = resnet50(weights=weights)
            model.fc = nn.Linear(model.fc.weight.shape[1], n_classes)

        elif model_name == 'vit':
            if pretrained:
                weights = ViT_B_16_Weights
            model = vit_b_16(weights=weights)
            model.heads.head = nn.Linear(model.heads.head.weight.shape[1], n_classes)

        elif model_name == 'biomedclip':

            model = biomed_class(n_classes)

            return model



    else:
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

        ############# 2D segmentation
        elif model_name == 'biomedclip':
            model = biomed_seg(n_classes)
            # print(model)
    return model