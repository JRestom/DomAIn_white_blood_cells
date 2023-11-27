import open_clip
from torch import nn

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