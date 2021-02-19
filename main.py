import argparse
import torch 
import random 
import numpy as np

import os
from utils import *
import logging
from shutil import copyfile

from torchvision import models
from torch.optim import *

from training import *
from dataset import *
from shapley import *
from dataset import *
from model import *

def main(args):
    
    train_config, train_config_path = get_config(args.train_config)
    model_config, model_config_path = get_config(args.model_config)
    set_all_seeds(train_config.seed)

    output_path = prepare_output_directory(train_config, model_config)
    copyfile(train_config_path, output_path + "/" + train_config_path.split("/")[-1])
    copyfile(model_config_path, output_path + "/" + model_config_path.split("/")[-1])

    if model_config.mask_inputs:
        distribution = (model_config.mask_mean, model_config.mask_std)
    else:
        distribution = None
    train_loader, test_loader, shapley_loader = get_loaders(train_config, distribution)
    model = build_model(train_config, train_loader, test_loader)

    criterion = lambda pred, labels: torch.softmax(pred, dim=-1).gather(dim=-1, index=labels.unsqueeze(-1))
    shapley_model = EstimateModel(model_config, train_config, criterion, distribution)
    outputs, loss = shapley_model.fit(model, shapley_loader)
    
    save_matrices(list(zip(*outputs)), loss, output_path)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", help="path to the trainer config", type=str)
    parser.add_argument("--model_config", help="path to model config", type=str)

    args = parser.parse_args()
    main(args)