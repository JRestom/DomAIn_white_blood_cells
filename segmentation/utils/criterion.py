from torch.nn import CrossEntropyLoss
from monai.losses import DiceLoss

def criterion(task):
    criterion = DiceLoss(
                            to_onehot_y=True, 
                            softmax=True                    
                            )
    return criterion