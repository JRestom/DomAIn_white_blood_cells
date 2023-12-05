# CV805 Final Project 

## Creating the environment
```sh
conda env create -f segmentation/environment.yml
```
## Running segmentation experiments
Determine the model, training strategy and other configurations in segmentation/configs.yaml. To train the segmentation model:

```sh
python segmentation/segmentation.py -p segmentation/configs.yaml
```
Three models are available for training:

* UNet 
* UNETR 
* SwinUNETR

Results will be saved in the segmentation/results files.
