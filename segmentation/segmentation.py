import torch
import torchvision
import torchvision.transforms as transforms
from avalanche.benchmarks.utils import make_avalanche_dataset
import numpy as np

from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.models import SimpleMLP
from avalanche.models import SimpleCNN
from torch.optim import SGD
from avalanche.training import Naive
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn


import sys


from utils.models import model_loader
from utils.strategies import strategy_loader
from utils.datasets import Dataset_creator
from utils.criterion import criterion
from utils.seg_evaluator import evaluator 
import argparse
import yaml
import pathlib
import time

from avalanche.benchmarks.utils import ImageFolder, DatasetFolder, FilelistDataset, AvalancheDataset
#print(torch. __version__ )
def main(args):
    path_to_config = pathlib.Path(args.path)
    with open(path_to_config) as f:
        config = yaml.safe_load(f)

    # Task
    task = config['task']
    # Dataset
    root = pathlib.Path(config['root'])
    ds_names = config['ds_names']
    #Model
    model_name = config['model_name'].lower()
    n_classes= config['n_classes']
    strategy= config['strategy']
    train_epochs= config['train_epochs']
   

    #-----------------Create datasets--------------------------------
    # #Create datasets
    datasets = list()
    for i, ds in enumerate(ds_names):
        train_ds, test_ds = Dataset_creator(
                                    root,
                                    task,
                                    ds_names[i],
                                    ).create_DS()
        datasets.append((train_ds, test_ds))





    #--------------Create benchmarks---------------------------------
    bm = dataset_benchmark(
        train_datasets=[datasets[i][0] for i in range(len(datasets))],
        test_datasets=[datasets[i][1] for i in range(len(datasets))],

    )
  



    #------------model-------------------------
    model = model_loader(model_name=model_name, task=task,n_classes=n_classes)
    total_params = sum(
	param.numel() for param in model.parameters()
    )

    trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad
    )
 



    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criter = criterion(task)

    cl_strategy = strategy_loader(strategy, model, optimizer, criter,train_mb_size=4, train_epochs=train_epochs, eval_mb_size=4, device='cuda')




    #----------------------- TRAINING LOOP-------------------------
    print('Starting experiment...')
    start = time.time()

    results =[]

    if strategy=='Joint':
        print('Using Joint strategy')
        experience = bm.train_stream
        print("Start of experience: ")


        cl_strategy.train(experience)
        print('Training completed')
      

        
            

        
    else:
        print(f'Using {strategy} strategy')

        for experience in bm.train_stream:
            print("Start of experience: ", experience.current_experience)

            cl_strategy.train(experience)
            print('Training completed')
    end = time.time()
    print(f"Total training time: {end - start} seconds")
    print("Evaluating ....")
    results_seg = evaluator(model, root)
            
        


    #----------------Saving the results-------------------------------
    # Define a file to write the results
    file_name = f"results/{strategy}_{model_name}_final.txt"
    # Open the file in write mode and write the results
    with open(file_name, "w") as file:
        file.write(f"Results for \"{strategy}\" strategy \n")
        file.write(f"Model: \"{model_name}\" \n")
        file.write(f"training time : {end - start} seconds")
    
        for center in list(results_seg.keys()):
            file.write('--------------------------------------------------------- \n')
            file.write(f"results on {center}: " + str(results_seg[center])+'\n')
    file.close()
              
        


    print(f"Results have been saved to {file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continual Learning Training Script')
    parser.add_argument("-p", "--path", type=str, required=True, help="path to the config file")
    args = parser.parse_args()
    main(args)