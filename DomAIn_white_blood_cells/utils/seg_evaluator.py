import monai
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference

import sys
import torch


def evaluator(model, test_datasets, names):
    results = list()
    device = torch.device("cuda")
    overall_dice = list()

    for i in range(len(test_datasets)):
        print(f"evaluating on {names[i]} Center")
        dit = {}
      
        total_dice = 0
        loader = DataLoader(test_datasets[i], batch_size=1, num_workers=4, shuffle=False)
        
        roi_size = (96,96,96)
        sw_batch_size = 1


        


        model.to(device)
        model.eval()
        with torch.no_grad():
        
            for batch in loader:
              
                im = batch[0].to(device)
                lbl = batch[1].to(device)
                preds = sliding_window_inference(im, roi_size, sw_batch_size, model)
                # print(preds.shape)

                dice_metric = DiceMetric(include_background=False, reduction="mean")
                dice_metric(preds, lbl)
                metric = dice_metric.aggregate().item()
                dice_metric.reset()

                overall_dice.append(metric)
        dit[names[i]]=overall_dice
        print(f"Dice Value on {names[i]} : {overall_dice[-1]}")
        results.append(dit)

    return results

    

