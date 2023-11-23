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
                ds_name,  
                train_size
                ):
        self.center = ds_name
        self.train_size = train_size
        self.task = task
        if task == 'classification':
            if ds_name == 'Raabin-WBC':
                self.data_path = os.path.join(root, 'Raabin-WBC/Train_Norm')
            elif ds_name == 'Matek-19':
                self.data_path = os.path.join(root, 'Matek-19/AML-Cytomorphology_LMU_Norm')
            elif ds_name == 'Acevedo-20':
                self.data_path = os.path.join(root, 'Acevedo-20/Train_Norm')

        else:   # for the segmentation we only have one dataset collection
            self.data_path = root



    def class_transforms(self, img):
        data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the desired size
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the pixel values
        ])
        return data_transforms(img)




    def class_img_loader(self, image_path):
        with open(image_path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def split_dataset(self, dataset, train_size):
        train_size = int(train_size * len(dataset))
        test_size = len(dataset) - train_size

        # Use random_split to create the training and testing subsets
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        return train_dataset, test_dataset


    def create_DS(self):
        if self.task == 'classification':
            dataset =  DatasetFolder(
                        root=self.data_path, 
                        extensions=self.CLS_EXTENSIONS,
                        transform=self.class_transforms, 
                        loader=self.class_img_loader
                        )
            train_ds, test_ds = self.split_dataset(
                                                dataset, 
                                                self.train_size
                                                )

        if self.task == 'segmentation':
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

