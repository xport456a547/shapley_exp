import torch 
import random 
import numpy as np

import json
from types import SimpleNamespace
import logging
import os
import time

def get_config(file):
    with open(file, 'r') as f:
        return json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))
    
def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def prepare_output_directory(train_config, model_config):

    path = "tmp/" + model_config.model 
    if train_config.from_pretrained:
        path += "_pretrained"
    else:
        path += "_scratch"  

    try:
        path += "_" + str(model_config.block_size) 
        path += "_" + str(model_config.stride) 

        if model_config.full_mask_only:
            path += "_full"
        elif model_config.circular_mask:
            path += "_circular"
        else:
            path += "_balanced"
    except: pass

    path += "_" + time.strftime("%Y%m%d%H%M%S") 
    os.makedirs(path)

    logging.basicConfig(
        filename=path + '/output.log', 
        level=logging.INFO, format="[%(asctime)s][%(levelname)s]%(message)s", 
        datefmt='%Y-%m-%d %H:%M:%S'
        )

    logging.info(f"Train config: {train_config}")
    logging.info(f"Train config: {model_config}")

    os.makedirs(path + "/img")
    os.makedirs(path + "/shapley")
        
    return path

def save_matrices(outputs, loss, path): 

    names = ["base", "processed", "segmentation", "shapley", "labels"]
    
    for name, output in zip(names, outputs):
        output = torch.cat(output, dim=0).cpu().numpy()

        logging.info(f"Saving {name} {output.shape}")
        with open(path + '/shapley/' + name + '.npy', 'wb') as f:
            np.save(f, output)

    logging.info(f"Saving loss {loss.shape}")
    with open(path + '/shapley/loss.npy', 'wb') as f:
        np.save(f, loss.cpu().numpy())


    