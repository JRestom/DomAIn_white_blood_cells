import torch
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
from PIL import Image
import numpy as np
from torch.utils.data import random_split

from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.generators import dataset_benchmark
from torch.optim import SGD

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn as nn

from avalanche.training.supervised import (
    Naive,
    Replay,
    EWC,
    JointTraining,
    SynapticIntelligence,
    LwF,
    GenerativeReplay
)
from avalanche.training.supervised.icarl import ICaRL
from avalanche.training.supervised.cumulative import Cumulative

import argparse
from models import biomed_class
import time
import os

#--------------Argument Parsing----------------------------
parser = argparse.ArgumentParser(description='Image Classification Training Script')
parser.add_argument('--architecture', type=str, required=True, help='Model architecture (e.g., "Resnet", "Vit")')
parser.add_argument('--weights', action='store_true', help='Use pre-trained weights if available')
parser.add_argument('--strategy', type=str, required=True, help='Training strategy (e.g., "Naive", "Joint", "EWC", "Replay")')
parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
parser.add_argument('--n_classes', type=int, required=True, help='Number of classes')
parser.add_argument('--save_dir', type=str, required=True, help='Path to save the results')
args = parser.parse_args()

architecture = args.architecture
use_pretrained_weights = args.weights
strategy = args.strategy
n_classes = args.n_classes
epochs = args.epochs
save_dir = args.save_dir

assert architecture in ['Resnet', 'Vit', 'Biomed_clip'], 'Architecture not supported'
assert strategy in ['Naive', 'Joint', 'Cumulative', 'EWC', 'Replay', 'SynapticIntelligence', 'LwF', 'GenerativeReplay'], 'Strategy not supported'
assert os.path.exists(save_dir), 'Save dir does not exist'

#--------------Data transformations----------------------------
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the pixel values
])


#--------------Custom loader----------------------------
def custom_loader(image_path):
    with open(image_path, 'rb') as f:
        return Image.open(f).convert('RGB')


#--------------File extensions ----------------------------
extensions = ('.png')


#--------------Create datasets----------------------------
dataset_raabin = DatasetFolder(root='/l/users/u21010212/cv805/datasets/Raabin-WBC/Train_Norm', extensions=extensions,transform=data_transforms, loader=custom_loader)
dataset_matek = DatasetFolder(root='/l/users/u21010212/cv805/datasets/Matek-19/AML-Cytomorphology_LMU_Norm', extensions=extensions,transform=data_transforms, loader=custom_loader)
dataset_acevedo = DatasetFolder(root='/l/users/u21010212/cv805/datasets/Acevedo-20/Train_Norm', extensions=extensions,transform=data_transforms, loader=custom_loader)


#--------------Train and Test subsets----------------------------
def split_dataset(dataset):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset


#--------------Train and Test splits----------------------------
train_raabin, test_raabin = split_dataset(dataset_raabin)
train_matek, test_matek = split_dataset(dataset_matek)
train_acevedo, test_acevedo = split_dataset(dataset_acevedo)


#--------------Create benchmarks---------------------------------
bm = dataset_benchmark(train_datasets=[train_raabin,train_matek,train_acevedo],
    test_datasets=[test_raabin, test_matek, test_acevedo],
    )


# --------------Model and Training Strategy Definition----------------------
if architecture == 'Resnet':
    if use_pretrained_weights:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        print('Weights loaded')
    else:
        model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.weight.shape[1], n_classes)

elif architecture == 'Vit':
    if use_pretrained_weights:
        model = vit_b_16(weights=ViT_B_16_Weights)
        print('Weights loaded')
    else:
        model = vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.weight.shape[1], n_classes)

elif architecture == 'Biomed_clip':
    model = biomed_class(n_classes)

optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = CrossEntropyLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

if strategy=='Naive':
    cl_strategy = Naive(model, optimizer, criterion,train_mb_size=32, train_epochs=epochs, eval_mb_size=32, device=device)

elif strategy=='Joint':
    cl_strategy = JointTraining(model,optimizer,criterion,train_mb_size=32,train_epochs=epochs,eval_mb_size=32,device=device)

elif strategy=='Cumulative':
    cl_strategy = Cumulative(model,optimizer,criterion,train_mb_size=32,train_epochs=epochs,eval_mb_size=32,device=device)

elif strategy=='EWC':
    cl_strategy = EWC(model,optimizer,criterion,train_mb_size=32,train_epochs=epochs,eval_mb_size=32,device=device, ewc_lambda=1.0e-1)

elif strategy=='LwF':
    cl_strategy = LwF(model,optimizer,criterion,train_mb_size=32,train_epochs=epochs,eval_mb_size=32,device=device, alpha=0.5, temperature=2.0)

elif strategy=='SynapticIntelligence':
    cl_strategy = SynapticIntelligence(model,optimizer,criterion,train_mb_size=32,train_epochs=epochs,eval_mb_size=32,device=device, si_lambda=1.0e-3)

elif strategy=='Replay':
    cl_strategy = Replay(model,optimizer,criterion,train_mb_size=32,train_epochs=epochs,eval_mb_size=32,device=device, mem_size=50)

elif strategy=='GenerativeReplay':
    cl_strategy = GenerativeReplay(model,optimizer,criterion,train_mb_size=32,train_epochs=epochs,eval_mb_size=32,device=device)




#----------------------- TRAINING LOOP-------------------------
print('Starting experiment...')

results = []
training_times = []

if strategy=='Joint':
    print('Using Joint strategy')
    experience = bm.train_stream
    print("Start of experience: ")

    start_time = time.time()
    cl_strategy.train(experience)
    end_time = time.time()

    training_duration = end_time - start_time
    training_times.append(training_duration)
    print(f'Training completed in {training_duration:.2f} seconds')

    print('Computing test accuracy on all datasets ')
    results.append(cl_strategy.eval(bm.test_stream))
    
else:
    print(f'Using {strategy} strategy')
    for experience in bm.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        start_time = time.time()
        cl_strategy.train(experience)
        end_time = time.time()

        training_duration = end_time - start_time
        training_times.append(training_duration)
        print(f'Training completed in {training_duration:.2f} seconds')

        print('Computing test accuracy on all datasets ')
        results.append(cl_strategy.eval(bm.test_stream)) 


#----------------Saving the results-------------------------------
file_name = f"{strategy}_{architecture}_{str(epochs)}_epochs.txt"
save_path = os.path.join(save_dir, file_name)
total_training_time = sum(training_times)

with open(save_path, "w") as file:
    for dic in results:
        file.write('--------------------------------------------------------- \n')
        file.write('results on raabin: ' + str(dic['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000'])+'\n')
        file.write('results on matek: ' + str(dic['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp001'])+'\n')
        file.write('results on acevedo: ' + str(dic['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp002'])+'\n')
    
    file.write("\nTraining Times\n")
    file.write(f"Total training time: {total_training_time:.2f} seconds\n")
    for i, time in enumerate(training_times):
        file.write(f"Training time for experience {i}: {time:.2f} seconds\n")

print(f"Results and training times have been saved to {save_path}")