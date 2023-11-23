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
import sys

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
    train_size = float(config['train_size'])
    #Model
    model_name = config['model_name'].lower()
    pretrained = config['pretrained']
    n_classes= config['n_classes']
   
    # center = config['center_name']
    
    


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
    datasets = list()
    for i, ds in enumerate(ds_names):
        train_ds, test_ds = Dataset_creator(
                                    root,
                                    task,
                                    ds_names[i],
                                    train_size
                                    ).create_DS()
        datasets.append((train_ds, test_ds))


    # Define the sizes for training and testing subsets

    def split_dataset(dataset):
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        # Use random_split to create the training and testing subsets
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        return train_dataset, test_dataset



    #--------------Create benchmarks---------------------------------
    bm = dataset_benchmark(
        train_datasets=[datasets[i][0] for i in range(len(datasets))],
        test_datasets=[datasets[i][1] for i in range(len(datasets))],

    )




    #------------Toy model-------------------------
    architecture = 'unet'
    weights=None
    strategy='Naive'
    n_classes = 5 

    model = model_loader(model_name=model_name, task=task,n_classes=n_classes)
    # print(model)

        


    optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.9)
    # criterion = CrossEntropyLoss()
    criter = criterion(task)

    cl_strategy = strategy_loader(strategy, model, optimizer, criter,train_mb_size=2, train_epochs=1, eval_mb_size=2, device='cuda')




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
        if task == 'classification':
            results.append(cl_strategy.eval(bm.test_stream))
        elif task == 'segmenatation':
            print(bm.test_stream)
            sys.exit()

        
    else:
        print(f'Using {strategy} strategy')

        for experience in bm.train_stream:
            print("Start of experience: ", experience.current_experience)
            # print("Current Classes: ", experience.classes_in_this_experience)

            cl_strategy.train(experience)
            print('Training completed')
            if task == 'classification':
                print('Computing test accuracy on all datasets ')
                results.append(cl_strategy.eval(bm.test_stream))

            elif task == 'segmentation':
                
                test_dss = [datasets[i][1] for i in range(len(datasets))]
                results_seg = evaluator(model, test_dss, ds_names)
                print(type(results_seg))
        


    #----------------Saving the results-------------------------------
    # Define a file to write the results
    file_name = f"{strategy}_{model_name}.txt"
    # Open the file in write mode and write the results
    with open(file_name, "w") as file:
        if task == 'classification':
            for dic in results:
                file.write('--------------------------------------------------------- \n')
                file.write('results on raabin: ' + str(dic['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000'])+'\n')
                file.write('results on matek: ' + str(dic['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp001'])+'\n')
                file.write('results on acevedo: ' + str(dic['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp002'])+'\n')
        elif task =='segmentation':
            for dic in results_seg:
                file.write('--------------------------------------------------------- \n')
                file.flush()
                for i, name in enumerate(ds_names):
                    file.write(f"results on {name}: " + str(dic[f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp00{i}"])+'\n')
                    file.flush()
        


    print(f"Results have been saved to {file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continual Learning Training Script')
    parser.add_argument("-p", "--path", type=str, required=True, help="path to the config file")
    args = parser.parse_args()
    main(args)