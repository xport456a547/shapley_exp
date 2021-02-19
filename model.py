import torch
import torch.nn as nn
from shapley import *

class EstimateModel():

    def __init__(self, model_config, train_config, criterion, noise_distribution):
        
        self.model_config = model_config
        self.train_config = train_config 

        self.criterion = criterion
        self.noise_distribution = noise_distribution

        self.model_name = model_config.model

    def fit(self, model, shapley_loader):

        if self.model_name == "deeplift":
            s = DeepLiftModel(
                model=model, 
                criterion=self.criterion, 
                noise_distribution=self.noise_distribution, 
                device=self.train_config.device
                )

            outputs = s.fit(shapley_loader, n_samples=self.train_config.test_size)

        elif self.model_name == "integratedgradient":
            s = IntegratedGradientsModel(
                model=model, 
                criterion=self.criterion, 
                noise_distribution=self.noise_distribution, 
                device=self.train_config.device
                )

            outputs = s.fit(shapley_loader, n_samples=self.train_config.test_size)

        elif self.model_name == "gradientshap":
            s = GradientShapModel(
                model=model, 
                criterion=self.criterion, 
                noise_distribution=self.noise_distribution, 
                device=self.train_config.device
                )

            outputs = s.fit(
                shapley_loader, 
                n_samples=self.train_config.test_size, 
                n_reestimations=self.model_config.n_reestimations,
                zero_baseline=self.model_config.zero_baseline
                )

        elif self.model_name == "deepliftshap":
            s = GradientShapModel(
                model=model, 
                criterion=self.criterion, 
                noise_distribution=self.noise_distribution, 
                device=self.train_config.device
                )

            outputs = s.fit(
                shapley_loader, 
                n_samples=self.train_config.test_size, 
                n_reestimations=self.model_config.n_reestimations, 
                zero_baseline=self.model_config.zero_baseline
                )

        elif self.model_name == "equalsurplus":
            s = EqualSurplusModel(
                model=model, 
                criterion=self.criterion, 
                noise_distribution=self.noise_distribution, 
                batch_size=self.train_config.test_batch_size, 
                device=self.train_config.device
                )

            mask = s.get_custom_block_mask(
                image_shape=(3, 224, 224), 
                block_size=(self.model_config.block_size, self.model_config.block_size), 
                stride=self.model_config.stride, 
                circular=self.model_config.circular_mask, 
                full_mask_only=self.model_config.full_mask_only
            )

            outputs = s.fit(shapley_loader, mask, n_samples=self.train_config.test_size, k_reestimate=self.model_config.k_reestimate)

        elif self.model_name == "equalxsurplus":
            s = EqualXSurplusModel(
                model=model, 
                criterion=self.criterion, 
                noise_distribution=self.noise_distribution, 
                batch_size=self.train_config.test_batch_size, 
                device=self.train_config.device
                )

            mask = s.get_custom_block_mask(
                image_shape=(3, 224, 224), 
                block_size=(self.model_config.block_size, self.model_config.block_size), 
                stride=self.model_config.stride, 
                circular=self.model_config.circular_mask, 
                full_mask_only=self.model_config.full_mask_only
            )

            outputs = s.fit(shapley_loader, mask, n_samples=self.train_config.test_size, k_reestimate=self.model_config.k_reestimate)

        segmentation_accuracy = s.segmentation_accuracy(outputs)
        logging.info(f"[{self.model_name}][Segmentation acc] {round(segmentation_accuracy, 4)}")
        
        loss = self.loss_difference_experiment(s, outputs)

        return outputs, loss

    def loss_difference_experiment(self, s, outputs):

        losses = []
        for i in range(19):
            t = round(0.05 + i*0.05, 3)
            features = s.extract_features(outputs, top=t)
            diff, std = s.loss_difference(features, self.criterion, reesample=self.train_config.loss_difference_reesample)
            logging.info(f"[{self.model_name}][top diff std] ({t}, {round(diff, 4)}, {round(std, 4)})")
            losses.append((t, diff, std))

        return torch.tensor(losses)
