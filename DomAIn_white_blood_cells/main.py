import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from avalanche.benchmarks.utils import make_avalanche_dataset
import numpy as np
from torch.utils.data import DataLoader, random_split

from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.models import SimpleMLP
from avalanche.models import SimpleCNN
from torch.optim import SGD
from avalanche.training import Naive
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn as nn

from utils.HECKTOR_DS import HECKTORDS
from monai.networks.nets import UNet
from monai.losses import DiceLoss


from avalanche.training import JointTraining
from avalanche.training import EWC
from avalanche.training import Replay
from avalanche.benchmarks.utils import ImageFolder, DatasetFolder, FilelistDataset, AvalancheDataset
#print(torch. __version__ )

#Define data transformations:
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to the desired size
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the pixel values
])




# Create a custom loader for your images
def custom_loader(image_path):
    with open(image_path, 'rb') as f:
        return Image.open(f).convert('RGB')
    
# Specify the file extensions 
extensions = ('.png')

# #Create datasets
dataset_raabin = DatasetFolder(root='/l/users/u21010212/cv805/datasets/Raabin-WBC/Train_Norm', extensions=extensions,transform=data_transforms, loader=custom_loader)
dataset_matek = DatasetFolder(root='/l/users/u21010212/cv805/datasets/Matek-19/AML-Cytomorphology_LMU_Norm', extensions=extensions,transform=data_transforms, loader=custom_loader)
dataset_acevedo = DatasetFolder(root='/l/users/u21010212/cv805/datasets/Acevedo-20/Train_Norm', extensions=extensions,transform=data_transforms, loader=custom_loader)

## Segmentation datasets
train_CHUP = HECKTORDS(root = '/l/users/roba.majzoub/hecktor2022', split='train', center='CHUP')
test_CHUP = HECKTORDS(root = '/l/users/roba.majzoub/hecktor2022', split='val', center='CHUP')

train_CHUS = HECKTORDS(root = '/l/users/roba.majzoub/hecktor2022', split='train', center='CHUS')
test_CHUS = HECKTORDS(root = '/l/users/roba.majzoub/hecktor2022', split='val', center='CHUS')


# Define the sizes for training and testing subsets

def split_dataset(dataset):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # Use random_split to create the training and testing subsets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


#--------------Train and Test splits----------------------------
train_raabin, test_raabin = split_dataset(dataset_raabin)
train_matek, test_matek = split_dataset(dataset_matek)
train_acevedo, test_acevedo = split_dataset(dataset_acevedo)



#--------------Create benchmarks---------------------------------

# bm = dataset_benchmark(train_datasets=[train_raabin,train_matek,train_acevedo],
#     test_datasets=[test_raabin, test_matek, test_acevedo],

bm = dataset_benchmark(train_datasets=[train_CHUP,train_CHUS],
    test_datasets=[test_CHUP, test_CHUS],
    
)

# print(f"{bm.train_stream.name} - len {len(bm.train_stream)}")
# print(f"{bm.test_stream.name} - len {len(bm.test_stream)}")

# exp = bm.train_stream[0]
# print(f"Experience {exp.current_experience}")
# print(f"Classes in this experience: {exp.classes_in_this_experience}")
# print(f"Previous classes: {exp.classes_seen_so_far}")
# print(f"Future classes: {exp.future_classes}")

# As always, we can iterate over it normally or with a pytorch
# data loader.
# For instance, we can use tqdm to add a progress bar.
# dataset = exp.dataset
# for i, data in enumerate(tqdm(dataset)):
#   pass
# print("\nNumber of examples:", i + 1)


#------------Toy model-------------------------
architecture = 'unet'
weights=None
strategy='Naive'
n_classes = 5 

if architecture=='Resnet':
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.weight.shape[1], n_classes)
    print(model)

elif architecture=='Vit':
    model = vit_b_16(weights=weights)
    print(model)
    model.heads.head = nn.Linear(model.heads.head.weight.shape[1], n_classes)
    print(model)

elif architecture=='unet':
    model = UNet(
                spatial_dims=3, 
                in_channels=1, 
                out_channels=3, 
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
    print(model)
    


optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
# criterion = CrossEntropyLoss()
criterion = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)

if strategy=='Naive':
    cl_strategy = Naive(model, optimizer, criterion,train_mb_size=2, train_epochs=1, eval_mb_size=2, device='cuda')

elif strategy=='Joint':
    cl_strategy = JointTraining(model,optimizer,criterion,train_mb_size=2,train_epochs=1,eval_mb_size=2,device='cuda')

elif strategy=='EWC':
    cl_strategy = EWC(model,optimizer,criterion,train_mb_size=2,train_epochs=1,eval_mb_size=2,device='cuda', ewc_lambda=1.0e-1)

elif strategy=='Replay':
    cl_strategy = Replay(model,optimizer,criterion,train_mb_size=2,train_epochs=1,eval_mb_size=2,device='cuda', mem_size=50)



#----------------------- TRAINING LOOP-------------------------
print('Starting experiment...')

results = []

if strategy=='Joint':
    print('Using Joint strategy')
    experience = bm.train_stream
    print("Start of experience: ")


    cl_strategy.train(experience)
    print('Training completed')

    print('Computing test accuracy on all datasets ')
    results.append(cl_strategy.eval(bm.test_stream))
    
else:
    print(f'Using {strategy} strategy')
    for experience in bm.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        cl_strategy.train(experience)
        print('Training completed')

        print('Computing test accuracy on all datasets ')
        results.append(cl_strategy.eval(bm.test_stream))
    


#----------------Saving the results-------------------------------
# Define a file to write the results
file_name = f"{strategy}_ViT.txt"
# Open the file in write mode and write the results
with open(file_name, "w") as file:
    for dic in results:
        file.write('--------------------------------------------------------- \n')
        file.write('results on raabin: ' + str(dic['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000'])+'\n')
        file.write('results on matek: ' + str(dic['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp001'])+'\n')
        # file.write('results on acevedo: ' + str(dic['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp002'])+'\n')
        
    


print(f"Results have been saved to {file_name}")