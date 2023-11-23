'''
Code from pytorch source code for TORCHVISION.DATASETS.CITYSCAPES
https://pytorch.org/vision/main/_modules/torchvision/datasets/cityscapes.html#Cityscapes
'''

import json
import os
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PIL import Image
import monai
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
  
from torchvision.transforms import ToTensor
import nibabel as nb
import json
# from .utils import extract_archive, iterable_to_str, verify_str_arg
from torchvision.datasets import VisionDataset



class HECKTORDS(VisionDataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Examples:

        Get semantic segmentation target

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')

            img, smnt = dataset[0]

        Get multiple targets

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])

            img, (inst, col, poly) = dataset[0]

        Validate on the "coarse" set

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                                 target_type='semantic')

            img, smnt = dataset[0]
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple(
        "CityscapesClass",
        ["name", "id", "train_id", "category", "category_id", "has_instances", "ignore_in_eval", "color"],
    )

    classes = [
        CityscapesClass("background", 0, 255, "background", 0, True, True, (0, 0, 0)),
        CityscapesClass("Primary Tumor", 1, 255, "Cancerous_cells", 1, False, False, (255, 0, 0)),
        CityscapesClass("Lymph Nodes", 2, 255, "Lymph_nodules", 0, False, False, (0, 255, 0)),
    ]
    

    def __init__(
        self,
        root: str,
        split: str = "train",
        center: str = "CHUP",
       
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
       
        CENTERS = ['CHUP', 'CHUV', 'CHUS', 'CHUM', 'HMR', 'HGJ', 'MDA']
        assert center in CENTERS, "Invalid center name!"

        all_data = self._load_json(path=os.path.join(self.root, f"splits/center_{center}.json"))

            
        self.images_dir = os.path.join(self.root, "imagesTr_cropped")
        self.targets_dir = os.path.join(self.root, "labelsTr_cropped")
        self.split = split
        self.images = []
        self.targets = []

    
        if split == 'train':
            self.data_list = all_data["train"]
        elif split == 'test':
            self.data_list = all_data["test"]
    
        valid_modes = ("train", "test")
        
       

        

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):

                raise RuntimeError(
                    "Dataset not found or incomplete. Please make sure all required folders for the"
                    ' specified "split" and "mode" are inside the "root" directory'
                )

        for im in self.data_list:

            self.images.append(os.path.join(self.images_dir, im))
            self.targets.append(os.path.join(self.targets_dir, im.split(".nii")[0]+"GT.nii.gz"))

    def transformations(self):
        if self.split == 'train':
            transforms = Compose(
                [
                    LoadImaged(keys=["image","label"], ensure_channel_first = True),
                    SpatialPadd(keys=["image","label"], spatial_size=(176,176,176), method='end'),
                    Orientationd(keys=["image","label"], axcodes="PLS"),
            
        
            
                    NormalizeIntensityd(keys=["image"], channel_wise=True),
                    
                    RandFlipd(
                        keys=["image","label"],
                        spatial_axis=[0],
                        prob=0.20,
                    ),
                    RandFlipd(
                        keys=["image","label"],
                        spatial_axis=[1],
                        prob=0.20,
                    ),
                    RandFlipd(
                        keys=["image","label"],
                        spatial_axis=[2],
                        prob=0.20,
                    ),
                    RandRotate90d(
                        keys=["image","label"],
                        prob=0.20,
                        max_k=3,
                    ),
                    FromMetaTensord(keys=["image","label"])
                    ]
                    )
        elif self.split== 'test':
          
            transforms = Compose(
                [
                    LoadImaged(keys=["image","label"], ensure_channel_first = True),
                    SpatialPadd(keys=["image","label"], spatial_size=(176,176,176), method='end'),
                    Orientationd(keys=["image","label"], axcodes="PLS"),

                    NormalizeIntensityd(keys=["image"], channel_wise=True),
                    FromMetaTensord(keys=["image","label"])
                    ]
                )
        return transforms
        

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise, target is a json object if target_type="polygon", else the image segmentation.
        """
        dit = {}
        dit['image'] = self.images[index]
        dit['label'] = self.targets[index]

        
        # print(dit)
        transformed = self.transformations()(dit)
        
        transformed['image'] = transformed['image'].float()
        transformed['label'] = transformed['label'].float()
        

        return transformed['image'], transformed['label']

    def __len__(self) -> int:
        return len(self.images)


    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path) as file:
            data = json.load(file)
        return data

   

# ds = HECKTORDS(root='/l/users/roba.majzoub/hecktor2022',split='train', center='CHUP')
# items = ds.__getitem__(0)
# for item in items:
#     print(item.shape)