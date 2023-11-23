from torch.nn import CrossEntropyLoss
from monai.losses import DiceLoss

def criterion(task):
    if task =='classification':
        criterion = CrossEntropyLoss()
    elif task == 'segmentation':
        criterion = DiceLoss(
                            include_background=False, 
                            to_onehot_y=True, 
                            softmax=False                    
                            )
    return criterion