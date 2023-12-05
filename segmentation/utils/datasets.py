from utils.HECKTOR_DS import HECKTORDS
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import random_split
import os

class Dataset_creator():
    CLS_EXTENSIONS = ['.png', '.jpg']
    def __init__(self,
                root, 
                task, 
                ds_name
                ):
        self.center = ds_name
        self.task = task
        
        self.data_path = root



    def create_DS(self):
        train_ds = HECKTORDS(
                            root=self.data_path, 
                            split='train', 
                            center=self.center
                            )
        test_ds = HECKTORDS(
                            root=self.data_path, 
                            split='test', 
                            center=self.center
                            )
        return train_ds, test_ds

