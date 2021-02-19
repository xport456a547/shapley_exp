from torchvision.datasets import * 
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import os
import shutil
from urllib.request import urlretrieve

import torch
import numpy as np
from scipy.misc import imread

import multiprocessing
from multiprocessing.pool import ThreadPool

from torchvision.transforms import transforms
import random 

import logging

def get_loaders(train_config, distribution):

    dataset = TrainingDataset(random_mask_distribution=distribution)
    train_data, test_data = dataset.get_dataset(train_config.test_size)

    logging.info(f"Train size {len(train_data)} / Test size {len(test_data)}")

    train_loader, test_loader, shapley_loader = dataset.tuple_to_loader(
        datasets=(train_data, test_data, test_data), 
        batch_sizes=(train_config.train_batch_size, train_config.test_batch_size, 1)
        )

    return train_loader, test_loader, shapley_loader

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class TrainingDataset():

    def __init__(self, random_mask_distribution=None):

        self.random_mask_distribution = random_mask_distribution
    
    def get_dataset(self, test_size=1024):
        
        return self.get_oxford_dataset(test_size)

    def tuple_to_loader(self, datasets, batch_sizes):

        if isinstance(batch_sizes, tuple) or isinstance(batch_sizes, list):
            assert len(datasets) == len(batch_sizes)
            return [DataLoader(dataset, batch_size=batch_size) for dataset, batch_size in zip(datasets, batch_sizes)]
            
        return DataLoader(datasets, batch_size=batch_sizes)

    def download_url(self, url, filepath):
        directory = os.path.dirname(os.path.abspath(filepath))
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(filepath):
            logging.info("Dataset already exists on the disk. Skipping download.")
            return

        with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(filepath)) as t:
            urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
            t.total = t.n

    def extract_archive(self, filepath):
        logging.info("Extracting archive")
        extract_dir = os.path.dirname(os.path.abspath(filepath))
        shutil.unpack_archive(filepath, extract_dir)

    def preprocess_mask(self, mask):
        mask = mask.astype(np.float32)
        mask[mask == 1.0] = 1.0
        mask[(mask == 2.0) | (mask == 3.0)] = 0.0
        return mask

    def get_image(self, d):

        transform = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            #transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_base = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224)
            ])

        dataset_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets/oxford-iiit-pet")
        data = d.split()
        
        name, label = data[0], int(data[2]) - 1

        img_base = torch.tensor(imread(os.path.join(dataset_directory, "images/" + name + ".jpg"), mode="RGB").transpose(2, 0, 1)).float()

        img = transform(img_base/255)
        img_base = transform_base(img_base/255)

        segmentation = imread(os.path.join(dataset_directory, "annotations/trimaps/" + name + ".png"), mode="L")
        segmentation = transform_base(torch.tensor(self.preprocess_mask(segmentation)).unsqueeze(0))


        if self.random_mask_distribution is not None:
            noise = torch.normal(
                mean=torch.zeros_like(img) + self.random_mask_distribution[0], 
                std=torch.zeros_like(img) + self.random_mask_distribution[1]
                )
            return (img_base, img * segmentation + (1 - segmentation) * noise, segmentation, torch.tensor(label))

        return (img_base, img, segmentation, torch.tensor(label))

        

    def get_oxford_dataset(self, test_size):
        
        dataset_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets/oxford-iiit-pet")
        
        filepath = os.path.join(dataset_directory, "images.tar.gz")
        """
        self.download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", filepath=filepath,
        )
        """
        self.extract_archive(filepath)

        filepath = os.path.join(dataset_directory, "annotations.tar.gz")
        """
        self.download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", filepath=filepath,
        )
        """
        self.extract_archive(filepath)

        annotation_file = open(os.path.join(dataset_directory, "annotations/list.txt")).readlines()[6:]


        logging.info("Processing dataset")
        if self.random_mask_distribution is not None:
            logging.info(f"Masking with N{self.random_mask_distribution} noise")
        else:
            logging.info(f"Skipping segmentation mask")


        with ThreadPool(processes=None) as pool:
            outputs = pool.map(self.get_image, annotation_file)

        random.seed(42)
        random.shuffle(outputs)
        return outputs[:-test_size], outputs[-test_size:]