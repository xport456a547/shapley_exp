import argparse
import os
import glob
import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import time 

def make_dirs():
    saving_path = "saved_imgs/" + time.strftime("%Y%m%d%H%M%S") 
    os.makedirs(saving_path)    
    os.makedirs(saving_path + "/top")
    os.makedirs(saving_path + "/segmented")
    os.makedirs(saving_path + "/loss")
    return saving_path

def get_config(file):
    with open(file, 'r') as f:
        return json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))

def get_paths(file):
    with open(file, "r", encoding="utf-8") as f:
        return f.read().splitlines()

def get_datasets(path):
    return np.load(path + "/shapley/base.npy"), np.load(path + "/shapley/processed.npy"), np.load(path + "/shapley/segmentation.npy")

def get_names(paths):
    names = []
    for path in paths:
        for file in glob.glob(path + "/*.json"):
            try: 
                names.append(get_config(file).model)
            except:
                pass
    return names

def get_shapley(paths):
    shapleys = []
    for path in paths:
        shapleys.append(np.load(path + "/shapley/shapley.npy"))
    return shapleys

def get_top(shapley, top):
    shapley = torch.tensor(shapley).reshape(-1)
    mask = torch.zeros_like(shapley)
    _, idx = torch.topk(shapley, k=top, dim=-1)
    mask = mask.scatter(dim=-1, index=idx, src=torch.ones_like(idx).float())
    return mask.reshape(1, 224, 224)

def filter_imgs(base, shapleys, top, n_samples, segmentation=None):
    outputs = []
    top = int(top * 224 * 224)
    if segmentation is not None:
        segmentation = np.mean(segmentation.reshape(-1, 224*224), axis=-1)

    for shapley in shapleys:
        filtered_imgs = []

        for i in range(n_samples):

            if segmentation is not None:
                top = int(segmentation[i] * 224 * 224)

            sample = get_top(shapley[i], top) * base[i]
            sample[sample == 0] = 0.15

            filtered_imgs.append(sample.numpy())
        outputs.append(filtered_imgs)
    return outputs

def plot(base, processed, filtered_imgs, names, n_samples, path):

    for n in range(config.n_samples):
        fig, axs = plt.subplots(nrows=1, ncols=2 + len(names), figsize=(3 * (len(names)), 6))
        for i, ax in enumerate(axs):
            if i == 0:
                ax.imshow(base[n].transpose(1, 2, 0))
                ax.set_title("initial")
                ax.axis("off")
            elif i == 1:
                ax.imshow(processed[n].transpose(1, 2, 0))
                ax.set_title("processed")
                ax.axis("off")
            else:
                ax.imshow(filtered_imgs[i-2][n].transpose(1, 2, 0))
                ax.set_title(str(names[i-2]))
                ax.axis("off")
        
        plt.savefig(path + "/" + str(n))
        plt.close()

def get_curves(paths):
    idx, ab, line = [], [], []
    for i, path in enumerate(paths):
        data = np.load(path + "/shapley/loss.npy")
        ab.append(data[:,1])
        line.append(data[:,3])
    return data[:,0], np.array(ab).T, np.array(line).T

def plot_lines(idx, ab, line, saving_path, names):
    fig = plt.figure(figsize=(8, 6))
    plt.plot(idx, ab)
    plt.legend(names, loc="best")
    plt.ylabel("Avg. abs. prob. difference")
    plt.xlabel("Top % pixels kept")
    plt.savefig(saving_path + "/loss/abs")

    fig = plt.figure(figsize=(8, 6))
    plt.plot(idx, line)
    plt.legend(names, loc="best")
    plt.ylabel("Avg. prob. difference")
    plt.xlabel("Top % pixels kept")
    plt.savefig(saving_path + "/loss/avg")
    

parser = argparse.ArgumentParser()
parser.add_argument("--plot_config", help="path to the trainer config", type=str, default="config/plot_config/plot_config.json")
parser.add_argument("--plot_list", help="path to the trainer config", type=str, default="config/plot_config/plot")
args = parser.parse_args()


config = get_config(args.plot_config)
paths = get_paths(args.plot_list)
base, processed, segmentation = get_datasets(paths[0])
names = get_names(paths)
shapleys = get_shapley(paths)
saving_path = make_dirs()

#IMGS
filtered_imgs = filter_imgs(base, shapleys, config.top, config.n_samples)
plot(base, processed, filtered_imgs, names, config.n_samples, saving_path + "/top")

filtered_imgs = filter_imgs(base, shapleys, config.top, config.n_samples, segmentation)
plot(base, processed, filtered_imgs, names, config.n_samples, saving_path + "/segmented")

#CURVES
idx, ab, line = get_curves(paths)
plot_lines(idx, ab, line, saving_path, names)


