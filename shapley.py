import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from math import sqrt
from torch.distributions.categorical import Categorical
from random import randrange
import logging 

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)



class BaseExplanationModel():

    def __init__(self, model, criterion=None, noise_distribution=(0, 0), batch_size=16, device="cuda"):
        
        self.model = model
        self.noise_distribution = noise_distribution
        self.batch_size = batch_size
        self.device = device

        if criterion is not None:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss(reduction="none")

    def initial_step(self, sample, return_grad=False):

        if return_grad:
            sample = torch.autograd.Variable(sample.to(self.device), requires_grad=True)

        outputs = self.model(sample.to(self.device))
        outputs = torch.softmax(outputs, dim=-1)

        label = outputs.argmax(dim=1)
        loss = self.criterion(outputs, label).flatten()

        if return_grad:
            loss.backward()
            return loss.detach(), label.detach(), sample.grad.data.clone().detach()
        return loss, label

    def step(self, sample, label):
        
        outputs = self.model(sample)
        loss = self.criterion(outputs, label)
        loss = loss.reshape(-1, 1, 1, 1).detach()
        return loss

    def get_noise(self, mask):
        noise = torch.normal(torch.zeros_like(mask) + self.noise_distribution[0], std=torch.zeros_like(mask) + self.noise_distribution[1])
        noise_mask = noise.clone()
        noise_mask[noise_mask != 0] = 1
        return noise, noise_mask

    def get_kernel_mask(self, image_shape=(3, 32, 32), kernel_size=1):
        
        C, H, W = image_shape
        num_pixels = H * W

        mask = torch.zeros(num_pixels, 1, num_pixels)
        idx = torch.arange(num_pixels).unsqueeze(-1).unsqueeze(-1)
        mask = torch.scatter(mask, dim=-1, index=idx, src=torch.ones_like(idx).float())

        mask = mask.reshape(num_pixels, 1, H, W)
        mask = nn.MaxPool2d(kernel_size, 1, padding=kernel_size//2)(mask)

        self.set_base_mask(C, H, W)
        print("mask_shape", mask.shape)
        return mask

    def get_block_mask(self, image_shape=(3, 32, 32), block_size=(1, 1)):
        
        transposer = nn.ConvTranspose2d(1, 1, kernel_size=block_size, stride=block_size, bias=False)
        transposer.weight = torch.nn.Parameter(torch.ones_like(transposer.weight), requires_grad=False)

        C, H, W = image_shape
        B, C = block_size

        assert H % B == 0
        assert W % C == 0

        num_blocks = H//B * W//C

        mask = torch.zeros(num_blocks, 1, num_blocks)
        idx = torch.arange(num_blocks).unsqueeze(-1).unsqueeze(-1)
        mask = torch.scatter(mask, dim=-1, index=idx, src=torch.ones_like(idx).float())

        mask = mask.reshape(num_blocks, 1, H//B, W//C)
        mask = transposer(mask)

        self.set_base_mask(C, H, W)

        m = mask.sum(dim=0).reshape(1, -1)
        print("mask_shape", mask.shape, m.max(dim=-1)[0], m.min(dim=-1)[0])
        return mask

    def get_custom_block_mask(self, image_shape=(3, 32, 32), block_size=(1, 1), stride=1, full_mask_only=False, circular=False):

        c, h, w = image_shape 
        n_h_chunks = h // stride 
        n_w_chunks = w // stride

        mask = torch.zeros(h, w)
        mask[:block_size[0], :block_size[1]] = 1

        outputs = []
        
        if full_mask_only:
            for i in range(n_h_chunks - block_size[0]//stride + 1):
                if i > 0:
                    mask = torch.roll(mask, dims=-2, shifts=stride)

                for j in range(n_w_chunks - block_size[1]//stride + 1):
                    if j > 0:
                        mask = torch.roll(mask, dims=-1, shifts=stride)

                    outputs.append(mask)

                mask = torch.roll(mask, dims=-1, shifts=block_size[0])
        
        elif circular:
            for i in range(n_h_chunks):
                if i > 0:
                    mask = torch.roll(mask, dims=-2, shifts=stride)

                for j in range(n_w_chunks):
                    if i != 0 or j != 0 :
                        mask = torch.roll(mask, dims=-1, shifts=stride)

                    outputs.append(mask)
        
        else:
            mask = torch.zeros(h, w)
            for i in range(n_h_chunks + block_size[0]//stride - 1):
                for j in range(n_w_chunks + block_size[1]//stride - 1):
                    new_mask = mask.clone()
                    i_0 = max(-block_size[0] + stride*(i + 1), 0)
                    i_1 = min(stride*(i + 1), h)

                    j_0 = max(-block_size[1] + stride*(j + 1), 0)
                    j_1 = min(stride*(j + 1), h)
                    new_mask[i_0:i_1, j_0:j_1] = 1
                    outputs.append(new_mask)
        
        mask = torch.stack(outputs).unsqueeze(1)
        print(mask.shape)
        return mask

    def set_base_mask(self, c, h, w):
        mask = torch.zeros(h*w, 1, h*w)
        idx = torch.arange(h*w).unsqueeze(-1).unsqueeze(-1)
        self.base_mask = torch.scatter(mask, dim=-1, index=idx, src=torch.ones_like(idx).float()).reshape((h*w, 1, h, w))

        print("base_mask_shape", self.base_mask.shape)

    def extract_features(self, data, top=0., use_segmentation=False, batch_size=32):

        imgs, masks, labels = [], [], []
        for d in data:
            _, img, __, shapley, label = d

            _, c, h, w = shapley.size()

            topk = int(__.mean()*w*h) if use_segmentation else int(top*w*h)
            shapley = shapley.reshape(-1)
            idx = shapley.argsort(dim=-1, descending=True)[:topk]

            mask = torch.zeros_like(shapley).scatter(dim=-1, index=idx, src=torch.ones_like(idx).float())
            masks.append(mask.reshape(1, 1, h, w))

            imgs.append(img)
            labels.append(label)

        return DataLoader(
            TensorDataset(torch.cat(imgs, dim=0), torch.cat(labels, dim=0), torch.cat(masks, dim=0)), 
            batch_size=batch_size
            )
    
    def segmentation_accuracy(self, data):
        
        accs = 0
        for d in data:
            _, img, __, shapley, label = d

            _, c, h, w = shapley.size()

            topk = int(__.mean()*w*h) 
            shapley = shapley.reshape(-1)
            idx = shapley.argsort(dim=-1, descending=True)[:topk]

            mask = torch.zeros_like(shapley).scatter(dim=-1, index=idx, src=torch.ones_like(idx).float())
            accs += (mask == __.reshape(-1)).float().mean()

        return float(accs) / len(data)
        
    def loss_difference(self, test_data, criterion, reesample=1):
        
        with torch.no_grad():
            self.model.eval()
            accuracy = []
            loss = []

            for i in range(reesample):
                for data in test_data:

                    inputs, labels, mask = data
                    inputs, labels, mask = inputs.to(self.device), labels.to(self.device), mask.to(self.device)
                    #noise = torch.normal(torch.zeros_like(mask) + self.noise_distribution[0], std=torch.zeros_like(mask) + self.noise_distribution[1])
                    noise, noise_mask = self.get_noise(mask)

                    true_outputs = self.model(inputs)
                    masked_outputs = self.model(inputs * mask + (1 - mask) * noise)

                    # mask valuable pixel
                    predicted = torch.argmax(masked_outputs, dim=-1)
                    #labels = torch.argmax(true_outputs, dim=-1)
                    accuracy.append((predicted == labels).flatten().float().cpu())

                    loss.append(criterion(true_outputs, labels).cpu() - criterion(masked_outputs, labels).cpu())

            loss = torch.cat(loss)
            return float(loss.mean()), float(loss.std())

    def minmax(self, values):
        n, c, h, w = values.size()
        values = values.reshape(1, -1)

        min_, _ = torch.min(values, dim=-1)
        max_, _ = torch.max(values, dim=-1)

        #std_ = (values - min_) / (max_ - min_)
        #values = std_ * (max_ - min_) + min_
        values = (values - min_) / (max_ - min_)
        return values.reshape(n, c, h, w)



class IntegratedGradientsModel(BaseExplanationModel):

    def __init__(self, model, criterion=None, noise_distribution=None, device="cuda"):

        super().__init__(model, criterion, noise_distribution, device)

    def fit(self, data_loader, n_samples=16):
        
        self.model.eval()
        outputs = []

        for i, data in enumerate(data_loader):
            _, img, __, ___ = data 

            shapley_score, label = self._fit(_, img)
            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu()))

            if i + 1 == n_samples:
                break
        return outputs

    def _fit(self, raw_img, sample):
        
        with torch.no_grad():
            ig = IntegratedGradients(self.model)
            sample = sample.to(self.device)
            label = self.model(sample).argmax(dim=-1)

            attributions = ig.attribute(sample, target=label, return_convergence_delta=False).float()
            #attributions = ig.attribute(sample, raw_img.to(self.device), target=label, return_convergence_delta=False)
            #attributions = ig.attribute(sample, sample, target=label, return_convergence_delta=False)
        return attributions.mean(dim=1, keepdim=True), label

class DeepLiftModel(BaseExplanationModel):

    def __init__(self, model, criterion=None, noise_distribution=None, device="cuda"):

        super().__init__(model, criterion, noise_distribution, device)

    def fit(self, data_loader, n_samples=16):
        
        self.model.eval()
        outputs = []

        for i, data in enumerate(data_loader):
            _, img, __, ___ = data 

            shapley_score, label = self._fit(_, img)
            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu()))

            if i + 1 == n_samples:
                break
        return outputs

    def _fit(self, raw_img, sample):
        
        with torch.no_grad():
            ig = DeepLift(self.model)
            sample = sample.to(self.device)
            label = self.model(sample).argmax(dim=-1)

            attributions = ig.attribute(sample, target=label, return_convergence_delta=False).float()
            #attributions = ig.attribute(sample, raw_img.to(self.device), target=label, return_convergence_delta=False)
            #attributions = ig.attribute(sample, sample, target=label, return_convergence_delta=False)

        return attributions.mean(dim=1, keepdim=True), label

class DeepLiftShapModel(BaseExplanationModel):

    def __init__(self, model, criterion=None, noise_distribution=None, device="cuda"):

        super().__init__(model, criterion, noise_distribution, device)

    def fit(self, data_loader, n_samples=16, n_reestimations=16, zero_baseline=True):
        
        self.model.eval()
        outputs = []

        for i, data in enumerate(data_loader):
            _, img, __, ___ = data 

            shapley_score, label = self._fit(_, img, n_reestimations, zero_baseline)
            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu()))

            if i + 1 == n_samples:
                break
        return outputs

    def _fit(self, raw_img, sample, n_reestimations, zero_baseline):
        
        with torch.no_grad():
            ig = DeepLiftShap(self.model)
            sample = sample.to(self.device)
            label = self.model(sample).argmax(dim=-1)

            if zero_baseline:
                attributions = ig.attribute(sample, baselines=torch.zeros_like(sample), stdevs=float(self.noise_distribution[-1]), n_samples=n_reestimations, target=label, return_convergence_delta=False).float()
            else:
                attributions = ig.attribute(sample, baselines=sample, stdevs=float(self.noise_distribution[-1]), target=label, n_samples=n_reestimations, return_convergence_delta=False).float()
        return attributions.mean(dim=1, keepdim=True), label

class GradientShapModel(BaseExplanationModel):

    def __init__(self, model, criterion=None, noise_distribution=None, device="cuda"):

        super().__init__(model, criterion, noise_distribution, device)

    def fit(self, data_loader, n_samples=16, n_reestimations=16, zero_baseline=True):
        
        self.model.eval()
        outputs = []

        for i, data in enumerate(data_loader):
            _, img, __, ___ = data 

            shapley_score, label = self._fit(_, img, n_reestimations, zero_baseline)
            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu()))

            if i + 1 == n_samples:
                break
        return outputs

    def _fit(self, raw_img, sample, n_reestimations, zero_baseline):
        
        with torch.no_grad():
            ig = GradientShap(self.model)
            sample = sample.to(self.device)
            label = self.model(sample).argmax(dim=-1)

            if zero_baseline:
                attributions = ig.attribute(sample, baselines=torch.zeros_like(sample), stdevs=float(self.noise_distribution[-1]), n_samples=n_reestimations, target=label, return_convergence_delta=False).float()
            else:
                attributions = ig.attribute(sample, baselines=sample, stdevs=float(self.noise_distribution[-1]), target=label, n_samples=n_reestimations, return_convergence_delta=False).float()
        return attributions.mean(dim=1, keepdim=True), label

class SampledXLossModel(BaseExplanationModel):

    def __init__(self, model, criterion=None, noise_distribution=(0, 0), batch_size=16, extra_context=1, device="cuda"):

        super().__init__(model, criterion, noise_distribution, batch_size, device)
        self.extra_context = extra_context

    def fit(self, data_loader, n_samples=16, n_reestimations=128):
        
        self.model.eval()
        outputs = []

        for i, data in enumerate(data_loader):
            _, img, __, ___ = data 

            shapley_score, label = self._fit(img, n_reestimations)
            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu()))

            if i + 1 == n_samples:
                break
        return outputs

    def _fit(self, sample, n_reestimations):
        
        _, c, h, w = sample.size()

        v_n, label, gradient = self.initial_step(sample, return_grad=True)

        gradient = gradient.norm(dim=1, keepdim=True)
        probs = torch.softmax(gradient.reshape(-1), dim=-1)
        
        count = 0
        values_inf = torch.zeros_like(gradient) + 100
        values_sup = torch.zeros_like(gradient) + 100
        mask_sum = torch.zeros_like(gradient) + 1e-6

        for _ in range(1):
            count = 0
            while count < n_reestimations:

                n = min(self.batch_size, n_reestimations - count)

                label = label.to(self.device)
                sample = sample.to(self.device).expand(n, -1, -1, -1)

                mask = self.sample(probs.expand(n, -1)).reshape(n, 1, h, w)
                #mask = self.sample((torch.ones_like(probs) / (h * w)).expand(n, -1)).reshape(n, 1, h, w)

                #v = (values_sup).sum(dim=0, keepdim=True).reshape(-1)
                #mask = self.sample(torch.softmax(v, dim=-1).expand(n, -1)).reshape(n, 1, h, w)

                #mask = self.sample2(values_inf, n)

                #extra_context = randrange(0, self.extra_context//2) * 2 + 1
                extra_context = self.extra_context

                if _ == 0:
                    mask2 = nn.MaxPool2d(kernel_size=self.extra_context, stride=1, padding=self.extra_context//2)(mask)
                else:
                    mask2 = nn.MaxPool2d(kernel_size=self.extra_context // (_*2) + 1, stride=1, padding=(self.extra_context // (_*4)))(mask)

                noise, noise_mask = self.get_noise(mask2)

                with torch.no_grad():
                    #values_sup += (self.step(sample * (1 - mask2) + mask2 * noise, label.expand(n)) * mask2).sum(dim=0, keepdim=True) #/ extra_context
                    #values_inf += (self.step(sample * mask2 + (1 - mask2) * noise, label.expand(n)) * mask2).sum(dim=0, keepdim=True) #/ extra_context
                    m = mask_sum.clone()
                    m[m < 1] = 1
                    values_sup = values_sup * m + (self.step(sample * (1 - mask2) + mask2 * noise, label.expand(n)) * mask2).sum(dim=0, keepdim=True)
                    values_inf = values_inf * m + (self.step(sample * mask2 + (1 - mask2) * noise, label.expand(n)) * mask2).sum(dim=0, keepdim=True)

                mask_sum += mask2.sum(dim=0, keepdim=True)

                m = mask_sum.clone()
                m[m < 1] = 1
                values_sup /= m
                values_inf /= m

                count += n

        
        values_inf -= 100 / mask_sum 
        values_sup -= 100 / mask_sum

        #values_inf /= mask_sum 
        #values_sup /= mask_sum 

        #print(mask_sum.flatten().min(), mask_sum.flatten().max())
        values_sup = v_n - values_sup

        sum_inf = values_inf.reshape(1, 1, -1).sum(dim=-1, keepdim=True).unsqueeze(-1)
        sum_sup = values_sup.reshape(1, 1, -1).sum(dim=-1, keepdim=True).unsqueeze(-1)
        probs = probs.reshape(1, 1, h, w)

        w_ = (v_n - sum_sup) / (sum_inf - sum_sup)

        #values_inf[values_inf == 0] = -10
        #values_sup[values_sup == 0] = -10

        values = values_inf * w_ + (1 - w_) * values_sup

        values = values.mean(dim=1, keepdim=True)
        print(w_)
        #values = self.minmax(values)
        #gradient = self.minmax(gradient)
        #return values + gradient, label

        #return values * probs, label
        return values, label
        #return self.minmax(values), label

    def sample(self, probs):

        mask = 0
        for i in range(2):
            sample = Categorical(probs=probs).sample()
            mask += nn.functional.one_hot(sample, num_classes=probs.size()[-1]).float()
        mask[mask > 1] = 1
        return mask

    def sample2(self, x, n):
        _, c, h, w = x.size()
        x = x.reshape(-1)
        _, idx = torch.topk(x, k=n, largest=True)
        output = torch.zeros(n, c*h*w, device=x.device)

        idx = idx.unsqueeze(-1)
        output = torch.scatter(output, dim=-1, index=idx, src=torch.ones_like(idx).float())
        return output.reshape(n, c, h, w)

class EqualSurplusModel(BaseExplanationModel):

    def __init__(self, model, criterion=None, noise_distribution=(0, 0), batch_size=16, extra_context=1, device="cuda"):

        super().__init__(model, criterion, noise_distribution, batch_size, device)
        self.extra_context = extra_context

    def fit(self, data_loader, mask, n_samples=16, k_reestimate=1):
        
        self.model.eval()
        outputs = []

        for i, data in enumerate(data_loader):
            _, img, __, ___ = data 

            shapley_score, label = self._fit(img, mask, k_reestimate)

            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu()))

            if i + 1 == n_samples:
                break
        return outputs

    def _fit(self, sample, mask, k_reestimate):

        n = mask.shape[0]

        v_n, predicted_label = self.initial_step(sample)

        loader = DataLoader(
            TensorDataset(sample.expand(mask.shape[0], -1, -1, -1), predicted_label.expand(n).cpu(), mask), 
            batch_size=self.batch_size
            ) 

        with torch.no_grad():
            inf_values = self.shapley_loop(loader, k_reestimate)
        scores = inf_values + (v_n - inf_values.sum()) / inf_values.shape[0]

        return (scores, predicted_label)

    def shapley_loop(self, data, k_reestimate):

        masks = 1e-6
        shapley_values = 0

        for _ in range(k_reestimate):
            for batch in data:
                inputs, labels, mask = batch
                inputs, labels, mask = inputs.to(self.device), labels.to(self.device), mask.to(self.device)
                
                mask2 = nn.MaxPool2d(kernel_size=self.extra_context, stride=1, padding=self.extra_context//2)(mask)
                noise = torch.normal(torch.zeros_like(mask2) + self.noise_distribution[0], std=torch.zeros_like(mask2) + self.noise_distribution[1])
                
                noise_mask = noise.clone()
                noise_mask[noise_mask != 0] = 1

                #inputs = inputs * mask + (1 - mask) * (inputs + noise) * noise_mask
                inputs = inputs * mask2 + (1 - mask2) * noise * noise_mask

                predictions = self.model(inputs)
                values = self.criterion(predictions, labels).reshape(-1, 1, 1, 1) * mask

                shapley_values += values.sum(dim=0, keepdim=True)
                masks += mask.sum(dim=0, keepdim=True)

        shapley_values /= masks
        return shapley_values.sum(dim=0, keepdim=True).reshape(1, *shapley_values.size()[-3:])

class EqualXSurplusModel(BaseExplanationModel):

    def __init__(self, model, criterion=None, noise_distribution=(0, 0), batch_size=16, extra_context=1, device="cuda"):

        super().__init__(model, criterion, noise_distribution, batch_size, device)
        self.extra_context = extra_context

    def fit(self, data_loader, mask, n_samples=16, k_reestimate=1):
        
        self.model.eval()
        outputs = []

        for i, data in enumerate(data_loader):
            _, img, __, ___ = data 

            shapley_score, label = self._fit(img, mask, k_reestimate)
            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu()))

            if i + 1 == n_samples:
                break
        return outputs

    def _fit(self, sample, mask, k_reestimate):

        n = mask.shape[0]

        v_n, predicted_label, gradient = self.initial_step(sample, return_grad=True)

        loader = DataLoader(
            TensorDataset(sample.expand(mask.shape[0], -1, -1, -1), predicted_label.expand(n).cpu(), mask), 
            batch_size=self.batch_size
            ) 

        with torch.no_grad():
            inf_values, sup_values = self.shapley_loop(loader, k_reestimate)
        
        sup_values = v_n - sup_values
        inf_sum = inf_values.sum()
        sup_sum = sup_values.sum()
        
        w = (v_n - sup_sum) / (inf_sum - sup_sum)
        scores = w * inf_values + (1 - w) * sup_values 
        
        #scores += gradient.norm(dim=1, keepdim=True)
        #scores = self.minmax(scores) + self.minmax(gradient.norm(dim=1, keepdim=True))
        #probs = torch.softmax(gradient.norm(dim=1, keepdim=True).flatten(), dim=-1).reshape(1, 1, *scores.size()[-2:])
        
        #scores += gradient.norm(dim=1, keepdim=True)
        print(w)
        #print(torch.min(a), torch.max(a))
        return (scores, predicted_label)

    def shapley_loop(self, data, k_reestimate):

        inputs, labels, mask = next(iter(data))

        masks = 1e-6
        shapley_values_inf = 0
        shapley_values_sup = 0

        for _ in range(k_reestimate):
            noise = torch.normal(torch.zeros(1, *inputs.size()[1:]) + self.noise_distribution[0], std=torch.zeros(1, *inputs.size()[1:]) + self.noise_distribution[1]).to(self.device)

            for batch in data:
                inputs, labels, mask = batch
                inputs, labels, mask = inputs.to(self.device), labels.to(self.device), mask.to(self.device)
                
                mask2 = nn.MaxPool2d(kernel_size=self.extra_context, stride=1, padding=self.extra_context//2)(mask)
                noise = torch.normal(torch.zeros_like(mask2) + self.noise_distribution[0], std=torch.zeros_like(mask2) + self.noise_distribution[1])

                predictions = self.model(inputs * mask2 + (1 - mask2) * noise)
                values = self.criterion(predictions, labels).reshape(-1, 1, 1, 1) * mask
                shapley_values_inf += values.sum(dim=0, keepdim=True)


                #mask2 = nn.MaxPool2d(kernel_size=self.extra_context, stride=1, padding=self.extra_context//2)(1 - mask)
                #predictions = self.model(inputs * mask2 + (1 - mask2) * noise)
                #predictions = self.model(inputs * (1 - mask2) + mask2 * noise)
                predictions = self.model(inputs * (1 - mask2) + mask2 * noise)

                values = self.criterion(predictions, labels).reshape(-1, 1, 1, 1) * mask
                shapley_values_sup += values.sum(dim=0, keepdim=True)

                masks += mask.sum(dim=0, keepdim=True)

        shapley_values_inf /= masks
        shapley_values_sup /= masks
        return shapley_values_inf, shapley_values_sup

class SampledEqualXSurplusModel(BaseExplanationModel):

    def __init__(self, model, criterion=None, noise_distribution=(0, 0), batch_size=16, extra_context=1, device="cuda"):

        super().__init__(model, criterion, noise_distribution, batch_size, device)
        self.extra_context = extra_context

    def fit(self, data_loader, mask, n_samples=16, k_reestimate=1):
        
        self.model.eval()
        outputs = []

        for i, data in enumerate(data_loader):
            _, img, __, ___ = data 

            shapley_score, label = self._fit(img, mask, k_reestimate)
            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu()))

            if i + 1 == n_samples:
                break
        return outputs

    def _fit(self, sample, mask, k_reestimate):

        n = mask.shape[0]

        v_n, predicted_label, gradient = self.initial_step(sample, return_grad=True)

        loader = DataLoader(
            TensorDataset(sample.expand(mask.shape[0], -1, -1, -1), predicted_label.expand(n).cpu(), mask), 
            batch_size=self.batch_size
            ) 

        with torch.no_grad():
            inf_values, sup_values = self.shapley_loop(loader, k_reestimate)
        
        sup_values = v_n - sup_values
        inf_sum = inf_values.sum()
        sup_sum = sup_values.sum()
        
        w = (v_n - sup_sum) / (inf_sum - sup_sum)
        scores = w * inf_values + (1 - w) * sup_values 
        
        #scores += gradient.norm(dim=1, keepdim=True)
        #scores = self.minmax(scores) + self.minmax(gradient.norm(dim=1, keepdim=True))
        #probs = torch.softmax(gradient.norm(dim=1, keepdim=True).flatten(), dim=-1).reshape(1, 1, *scores.size()[-2:])
        
        #scores += gradient.norm(dim=1, keepdim=True)
        print(w)
        #print(torch.min(a), torch.max(a))
        return (scores, predicted_label)

    def shapley_loop(self, data, k_reestimate):

        masks = 1e-6
        shapley_values_inf = 0
        shapley_values_sup = 0

        for _ in range(k_reestimate):
            for batch in data:
                inputs, labels, mask = batch
                inputs, labels, mask = inputs.to(self.device), labels.to(self.device), mask.to(self.device)
                
                mask2 = nn.MaxPool2d(kernel_size=self.extra_context, stride=1, padding=self.extra_context//2)(mask)
                noise = torch.normal(torch.zeros_like(mask2) + self.noise_distribution[0], std=torch.zeros_like(mask2) + self.noise_distribution[1])

                predictions = self.model(inputs * mask2 + (1 - mask2) * noise)
                values = self.criterion(predictions, labels).reshape(-1, 1, 1, 1) * mask
                shapley_values_inf += values.sum(dim=0, keepdim=True)


                #mask2 = nn.MaxPool2d(kernel_size=self.extra_context, stride=1, padding=self.extra_context//2)(1 - mask)
                #predictions = self.model(inputs * mask2 + (1 - mask2) * noise)
                #predictions = self.model(inputs * (1 - mask2) + mask2 * noise)
                predictions = self.model(inputs * (1 - mask2) + mask2 * noise)

                values = self.criterion(predictions, labels).reshape(-1, 1, 1, 1) * mask
                shapley_values_sup += values.sum(dim=0, keepdim=True)

                masks += mask.sum(dim=0, keepdim=True)

        shapley_values_inf /= masks
        shapley_values_sup /= masks
        return shapley_values_inf, shapley_values_sup

class FastEqualXSurplusModel(BaseExplanationModel):

    def __init__(self, model, criterion=None, noise_distribution=(0, 0), batch_size=16, extra_context=1, device="cuda"):

        super().__init__(model, criterion, noise_distribution, batch_size, device)
        self.extra_context = extra_context

    def fit(self, data_loader, mask, n_samples=16, k_reestimate=1):
        
        self.model.eval()
        outputs = []

        for i, data in enumerate(data_loader):
            _, img, __, ___ = data 

            shapley_score, label = self._fit(img, mask, k_reestimate)
            outputs.append((_, img.cpu(), __, shapley_score.detach().cpu(), label.detach().cpu()))

            if i + 1 == n_samples:
                break
        return outputs

    def _fit(self, sample, mask, k_reestimate):

        n = mask.shape[0]

        v_n, predicted_label, gradient = self.initial_step(sample, return_grad=True)

        loader = DataLoader(
            TensorDataset(sample.expand(mask.shape[0], -1, -1, -1), gradient.cpu().expand(mask.shape[0], -1, -1, -1), predicted_label.expand(n).cpu(), mask), 
            batch_size=self.batch_size
            ) 

        with torch.no_grad():
            inf_values, sup_values = self.shapley_loop(loader, k_reestimate)
        
        v_n = (sample*gradient.cpu()).sum()
        sup_values = v_n - sup_values
        inf_sum = inf_values.sum()
        sup_sum = sup_values.sum()
        
        w = (v_n - sup_sum) / (inf_sum - sup_sum)
        scores = w * inf_values + (1 - w) * sup_values 
        
        #scores += gradient.norm(dim=1, keepdim=True)
        #scores = self.minmax(scores) + self.minmax(gradient.norm(dim=1, keepdim=True))
        #probs = torch.softmax(gradient.norm(dim=1, keepdim=True).flatten(), dim=-1).reshape(1, 1, *scores.size()[-2:])
        
        #scores += gradient.norm(dim=1, keepdim=True)
        print(w)
        #print(torch.min(a), torch.max(a))
        return (scores, predicted_label)

    def shapley_loop(self, data, k_reestimate):
        
        inputs, gradient, labels, mask = next(iter(data))

        masks = 1e-6
        shapley_values_inf = 0
        shapley_values_sup = 0

        for _ in range(k_reestimate):
            noise = torch.normal(torch.zeros(1, *inputs.size()[1:]) + self.noise_distribution[0], std=torch.zeros(1, *inputs.size()[1:]) + self.noise_distribution[1]).to(self.device)
            
            for batch in data:
                inputs, gradient, labels, mask = batch
                inputs, gradient, labels, mask = inputs.to(self.device), gradient.to(self.device), labels.to(self.device), mask.to(self.device)

                gradient = gradient.mean(dim=1, keepdim=True)
                
                mask2 = nn.MaxPool2d(kernel_size=self.extra_context, stride=1, padding=self.extra_context//2)(mask)

                n = inputs.size()[0]
                values = (inputs * gradient * mask2 + (1 - mask2) * (noise - inputs) * gradient).reshape(n, -1).sum(dim=-1).reshape(n, 1, 1, 1) * mask
                shapley_values_inf += values.mean(dim=0, keepdim=True)

                values = (inputs * gradient * (1 - mask2) + mask2 * (noise - inputs) * gradient).reshape(n, -1).sum(dim=-1).reshape(n, 1, 1, 1) * mask
                shapley_values_sup += values.sum(dim=0, keepdim=True)

                masks += mask.sum(dim=0, keepdim=True)

        shapley_values_inf /= masks
        shapley_values_sup /= masks
        return shapley_values_inf, shapley_values_sup
