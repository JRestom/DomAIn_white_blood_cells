import monai
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch

from monai.transforms import (
    AsDiscrete,
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    MapTransform,
    SpatialPadd,
    ConcatItemsd,
    NormalizeIntensityd, 
    FromMetaTensord
)
from monai.data import (
    DataLoader,
    Dataset,
    decollate_batch,
)

import os
import sys
import torch
import sys
import json
import numpy as np





def evaluator(model, root):

    CENTERS = ['CHUV', 'CHUS', 'CHUM', 'HMR', 'HGJ', 'MDA']
    images_dir = os.path.join(root, "imagesTr_cropped")
    targets_dir = os.path.join(root, "labelsTr_cropped")

    eval_data = []

    val_transforms = Compose(
            [
                LoadImaged(keys=["image","label"], ensure_channel_first = True),
                SpatialPadd(keys=["image","label"], spatial_size=(176,176,176), method='end'),
                Orientationd(keys=["image","label"], axcodes="PLS"),

                NormalizeIntensityd(keys=["image"], channel_wise=True),
                FromMetaTensord(keys=["image","label"])
                ]
            )

    dit = {}
    for center in CENTERS:
        all_data = load_json(path=os.path.join(root, f"splits/center_{center}.json"))
        val_data = all_data['test']
        for d in val_data:
            eval_data.append(d)

        
        test_files = [{'image':os.path.join(images_dir,eval_data[i]), 'label':os.path.join(targets_dir, eval_data[i].split(".")[0]+"GT.nii.gz")} for i in range(len(eval_data))]
        
        
        
        val_ds = Dataset(data=test_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, shuffle= False)

        post_pred = Compose([AsDiscrete(argmax=True, to_onehot=3)])
        post_label = Compose([AsDiscrete(to_onehot=3)])
        
        dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")

        
        device = torch.device("cuda")
        metric_values = []
        metric_values_tumor = []
        metric_values_lymph = []

        model.eval()
        dit1 = {}
        with torch.no_grad():
            for batch in val_loader:
                
                val_inputs = batch['image'].to(device)
                val_labels = batch['label'].to(device)
                

                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                dice_metric_batch(y_pred=val_output_convert, y=val_labels_convert)

            mean_dice_val = dice_metric.aggregate().item()
            metric_batch_val = dice_metric_batch.aggregate()

            metric_tumor = metric_batch_val[0].item()
            metric_lymph = metric_batch_val[1].item()

            dice_metric.reset()
            dice_metric_batch.reset()
        
            metric_values.append(mean_dice_val)
            metric_values_tumor.append(metric_tumor)
            metric_values_lymph.append(metric_lymph)
            dit1["average"] = metric_values
            dit1["tumor_dice"] = metric_values_tumor
            dit1["lymph_dice"] = metric_values_lymph

            dit[center] = dit1
            print(dit1)

    return dit

    
def load_json(path: str):
        with open(path) as file:
            data = json.load(file)
        return data

