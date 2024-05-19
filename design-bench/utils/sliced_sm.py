import os
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torch.utils.data import DataLoader, Subset
import shutil
import argparse
import tensorboardX
import logging

import pdb


config = {
    'training': {
        'batch_size': 128,
        'n_epochs': 500000,
        'n_iters': 50001,
        'ngpu': 1,
        'noise_std': 0.01,
        'algo': "ssm",
        'snapshot_freq': 5000,
    },
    'data': {
        ### mnist
        # 'dataset': "MNIST",
        # 'image_size': 16,
        # 'channels': 1,
        # 'logit_transform': False,
        ## celeba
        'dataset': "CIFAR10",
        'image_size': 32,
        'channels': 3,
        'logit_transform': False,
    },
    'model': {
        'n_particles': 1,
        'lam': 10,
        'z_dim': 100,
        'nef': 32,
        'ndf': 32,
    },
    'optim': {
        'weight_decay': 0.000,
        'optimizer': "Adam",
        'lr': 0.001,
        'beta1': 0.9,
    },
}


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def sliced_score_estimation_vr(score_net, samples: torch.Tensor, n_particles=1):
    """
    Be careful if the shape of samples is not B x x_dim!!!!
    """
    # samples [B, C, D, D]
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    # [1, B, C, D, D] -> [1, B, C, D, D] -> [B, C, D, D]
    dup_samples.requires_grad_(True) # dup_samples [B, C, D, D]
    vectors = torch.randn_like(dup_samples) # [B, C, D, D]

    grad1 = score_net(dup_samples)  # [B, C, D, D]
    gradv = torch.sum(grad1 * vectors)  # [,]
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]  # [B, C, D, D]

    grad1 = grad1.view(dup_samples.shape[0], -1)  # [B, C x D x D]
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.  # [B, C x D x D] -> [B,]

    loss2 = torch.sum((vectors * grad2).view(dup_samples.shape[0], -1), dim=-1)  # [B, C, D, D] -> [B, C x D x D] -> [B,]

    loss1 = loss1.view(n_particles, -1).mean(dim=0)  # [1, B] -> [B,]
    loss2 = loss2.view(n_particles, -1).mean(dim=0)  # [1, B] -> [B,]

    loss = loss1 + loss2  # [B,]
    return loss.mean(), loss1.mean(), loss2.mean()


def sliced_score_matching_vr(energy_net, samples: torch.Tensor, n_particles=1):
    """
    The shape of samples is [B, D] by default.
    """
    # samples [B, D]
    dup_samples = samples.unsqueeze(0).expand(n_particles, *samples.shape).contiguous().view(-1, *samples.shape[1:])
    # [1, B, D] -> [1, B, D] -> [B, D]
    dup_samples.requires_grad_(True) # dup_samples [B, D]
    vectors = torch.randn_like(dup_samples) # [B, D]

    logp = -energy_net(dup_samples).sum()  # [B,] -> [,]
    grad1 = autograd.grad(logp, dup_samples, create_graph=True)[0]  # [B, D]
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.  # [B, D] -> [B,]
    gradv = torch.sum(grad1 * vectors)  # [,]
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]  # [B, D]
    loss2 = torch.sum(vectors * grad2, dim=-1)  # [B,]

    loss1 = loss1.view(n_particles, -1).mean(dim=0)  # [1, B] -> [B,]
    loss2 = loss2.view(n_particles, -1).mean(dim=0)  # [1, B] -> [B,]

    loss = loss1 + loss2  # [B,]
    return loss.mean(), loss1.mean(), loss2.mean()


def dsm_score_estimation(scorenet, samples: torch.Tensor, sigma=0.01):
    perturbed_samples = samples + torch.randn_like(samples) * sigma
    target = - 1 / (sigma ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)

    return loss


def dsm(energy_net, samples, sigma=1):
    samples.requires_grad_(True)
    vector = torch.randn_like(samples) * sigma
    perturbed_inputs = samples + vector
    logp = -energy_net(perturbed_inputs)
    dlogp = sigma ** 2 * autograd.grad(logp.sum(), perturbed_inputs, create_graph=True)[0]
    kernel = vector
    loss = torch.norm(dlogp + kernel, dim=-1) ** 2
    loss = loss.mean() / 2.

    return loss


class MLPScore(nn.Module):
    def __init__(self, input_shape, act):
        super().__init__()
        input_size = np.prod(input_shape)

        def get_act():
            if act == 'relu':
                return nn.ReLU()
            elif act == 'softplus':
                return nn.Softplus()
            elif act == 'elu':
                return nn.ELU()
            elif act == 'leakyrelu':
                return nn.LeakyReLU(0.2)
            
        self.config = config
        self.main = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.LayerNorm(2048),
            get_act(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            get_act(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            get_act(),
            nn.Linear(2048, input_size),
        )

    def forward(self, x: torch.Tensor, ngpu=1):
        x = x.view(x.shape[0], -1)
        if x.is_cuda and ngpu > 1:
            score = nn.parallel.data_parallel(
                self.main, x, list(range(ngpu)))
        else:
            score = self.main(x)

        return score.view(x.shape[0], -1)


class ConvResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, resize=False, act='relu'):
        super().__init__()
        self.resize = resize

        def get_act():
            if act == 'relu':
                return nn.ReLU(inplace=True)
            elif act == 'softplus':
                return nn.Softplus()
            elif act == 'elu':
                return nn.ELU()
            elif act == 'leakyrelu':
                return nn.LeakyReLU(0.2, inplace=True)

        if not resize:
            self.main = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel),
                get_act(),
                nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel)
            )
        else:
            self.main = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1),
                nn.GroupNorm(8, out_channel),
                get_act(),
                nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel)
            )
            self.residual = nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1)

        self.final_act = get_act()

    def forward(self, inputs):
        if not self.resize:
            h = self.main(inputs)
            h += inputs
        else:
            h = self.main(inputs)
            res = self.residual(inputs)
            h += res
        return self.final_act(h)


class DeconvResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, resize=False, act='relu'):
        super().__init__()
        self.resize = resize

        def get_act():
            if act == 'relu':
                return nn.ReLU(inplace=True)
            elif act == 'softplus':
                return nn.Softplus()
            elif act == 'elu':
                return nn.ELU()
            elif act == 'leakyrelu':
                return nn.LeakyReLU(0.2, True)

        if not resize:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel),
                get_act(),
                nn.ConvTranspose2d(out_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel)
            )
        else:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel),
                get_act(),
                nn.ConvTranspose2d(out_channel, out_channel, 3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(8, out_channel)
            )
            self.residual = nn.ConvTranspose2d(in_channel, out_channel, 3, stride=2, padding=1, output_padding=1)

        self.final_act = get_act()

    def forward(self, inputs):
        if not self.resize:
            h = self.main(inputs)
            h += inputs
        else:
            h = self.main(inputs)
            res = self.residual(inputs)
            h += res
        return self.final_act(h)
    

class ResScore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nef = config.model.nef
        self.ndf = config.model.ndf
        act = 'elu'
        self.convs = nn.Sequential(
            nn.Conv2d(3, self.nef, 3, 1, 1),
            ConvResBlock(self.nef, self.nef, act=act),
            ConvResBlock(self.nef, 2 * self.nef, resize=True, act=act),
            ConvResBlock(2 * self.nef, 2 * self.nef, act=act),
            # ConvResBlock(2 * self.nef, 2 * self.nef, resize=True, act=act),
            # ConvResBlock(2 * self.nef, 2 * self.nef, act=act),
            ConvResBlock(2 * self.nef, 4 * self.nef, resize=True, act=act),
            ConvResBlock(4 * self.nef, 4 * self.nef, act=act),
        )

        self.deconvs = nn.Sequential(
            # DeconvResBlock(2 * self.ndf, 2 * self.ndf, act=act),
            # DeconvResBlock(2 * self.ndf, 2 * self.ndf, resize=True, act=act),
            DeconvResBlock(4 * self.ndf, 4 * self.ndf, act=act),
            DeconvResBlock(4 * self.ndf, 2 * self.ndf, resize=True, act=act),
            DeconvResBlock(2 * self.ndf, 2 * self.ndf, act=act),
            DeconvResBlock(2 * self.ndf, self.ndf, resize=True, act=act),
            DeconvResBlock(self.ndf, self.ndf, act=act),
            nn.Conv2d(self.ndf, 3, 3, 1, 1)
        )

    def forward(self, x):
        x = 2 * x - 1.
        res = self.deconvs(self.convs(x))
        return res


class ResEnergy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nef = config.model.nef
        self.ndf = config.model.ndf
        act = 'softplus'
        self.convs = nn.Sequential(
            # nn.Conv2d(1, self.nef, 3, 1, 1),
            nn.Conv2d(3, self.nef, 3, 1, 1),
            ConvResBlock(self.nef, self.nef, act=act),
            ConvResBlock(self.nef, 2 * self.nef, resize=True, act=act),
            ConvResBlock(2 * self.nef, 2 * self.nef, act=act),
            ConvResBlock(2 * self.nef, 4 * self.nef, resize=True, act=act),
            ConvResBlock(4 * self.nef, 4 * self.nef, act=act)
        )

    def forward(self, x):
        x = 2 * x - 1.
        res = self.convs(x)
        res = res.view(res.shape[0], -1).mean(dim=-1)
        return res
    

class MLPEnergy(nn.Module):
    def __init__(self, input_shape, act):
        super().__init__()
        input_size = np.prod(input_shape)

        def get_act():
            if act == 'relu':
                return nn.ReLU()
            elif act == 'softplus':
                return nn.Softplus()
            elif act == 'elu':
                return nn.ELU()
            elif act == 'leakyrelu':
                return nn.LeakyReLU(0.2)
            
        self.config = config
        self.main = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.LayerNorm(2048),
            get_act(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            get_act(),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            get_act(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            get_act(),

            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 1),
        )

        self._mean = None
        self._std = None

    def forward(self, x: torch.Tensor, ngpu=1):
        x = x.view(x.shape[0], -1)
        if x.is_cuda and ngpu > 1:
            energy = nn.parallel.data_parallel(
                self.main, x, list(range(ngpu)))
        else:
            energy = self.main(x)

        return energy.view(x.shape[0], -1).mean(dim=-1)
    
    def log_prob(self, x: torch.Tensor):
        """
        :param x: [B, D]
        :return logp: unscaled logp \propto -E [B, 1]
        """
        logp = -self.forward(x).unsqueeze(1)
        return logp
    
    def scaled_log_prob(self, x: torch.Tensor):
        """
        :param x: [B, D]
        :return logp: scaled logp [B, 1]
        """
        assert (self._mean and self._std), "Run prepare first!"
        logp = self.log_prob(x)
        scaled_logp = (logp - self._mean) / self._std
        return scaled_logp
    
    def score(self, x: torch.Tensor, flag=True):
        """
        :param x: [B, D]
        :return dlogp/dx: [B, D]
        """
        x.requires_grad_(True)
        logp = self.log_prob(x)
        if flag:
            grad = autograd.grad(logp.sum(), x)[0]
        else:
            logp.backward(torch.ones_like(logp))
            grad = x.grad
        return grad # [B, D]

    def scaled_score(self, x: torch.Tensor, flag=True):
        x.requires_grad_(True)
        scaled_logp = self.scaled_log_prob(x)
        if flag:
            grad = autograd.grad(scaled_logp.sum(), x)[0]
        else:
            scaled_logp.backward(torch.ones_like(scaled_logp))
            grad = x.grad
        return grad # [B, D]
    
    def prepare(self, x, is_dataloader=True):
        if is_dataloader:
            # iterate through the entire dataset a first time
            samples = x_mean = 0
            for i, (x_batch, y_batch) in enumerate(x):

                # calculate how many samples are actually in the current batch
                batch_size = x_batch.shape[0]

                # update the running mean using dynamic programming
                with torch.no_grad():
                    x_mean = x_mean * (samples / (samples + batch_size)) + \
                        torch.sum(self.log_prob(x_batch),
                            dim=0, keepdim=True) / (samples + batch_size)

                # update the number of samples used in the calculation
                samples += batch_size

            # iterate through the entire dataset a second time
            samples = x_variance = 0
            for i, (x_batch, y_batch) in enumerate(x):

                # calculate how many samples are actually in the current batch
                batch_size = x_batch.shape[0]

                # update the running variance using dynamic programming
                with torch.no_grad():
                    x_variance = x_variance * (samples / (samples + batch_size)) + \
                        torch.sum(torch.square(self.log_prob(x_batch) - x_mean),
                            dim=0, keepdim=True) / (samples + batch_size)

                # update the number of samples used in the calculation
                samples += batch_size

            # expose the calculated mean and standard deviation
            self._mean = x_mean.squeeze()
            self._std = torch.sqrt(x_variance).squeeze()
        else:
            n_samples = min(x.shape[0], 4096)
            random_state = np.random.get_state()
            np.random.seed(2024)
            n_indices = np.random.choice(np.arange(x.shape[0]), size=n_samples, replace=False)
            np.random.set_state(random_state)

            x = x[n_indices]
            self._mean = self.log_prob(x).mean()
            self._std = self.log_prob(x).std()
            pdb.set_trace()
        
        return self._mean, self._std
    

class ScoreNetRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999))
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    def train(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])

        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=transform)
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=False, download=True,
                                   transform=transform)
        elif self.config.data.dataset == 'MNIST':
            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=transform)
            num_items = len(dataset)
            indices = list(range(num_items))
            random_state = np.random.get_state()
            np.random.seed(2019)
            np.random.shuffle(indices)
            np.random.set_state(random_state)
            train_indices, test_indices = indices[:int(num_items * 0.8)], indices[int(num_items * 0.8):]
            test_dataset = Subset(dataset, test_indices)
            dataset = Subset(dataset, train_indices)

        elif self.config.data.dataset == 'CELEBA':
            dataset = ImageFolder(root=os.path.join(self.args.run, 'datasets', 'celeba'),
                                  transform=transforms.Compose([
                                      transforms.CenterCrop(140),
                                      transforms.Resize(self.config.data.image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ]))
            num_items = len(dataset)
            indices = list(range(num_items))
            random_state = np.random.get_state()
            np.random.seed(2019)
            np.random.shuffle(indices)
            np.random.set_state(random_state)
            train_indices, test_indices = indices[:int(num_items * 0.7)], indices[
                                                                          int(num_items * 0.7):int(num_items * 0.8)]
            test_dataset = Subset(dataset, test_indices)
            dataset = Subset(dataset, train_indices)

        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True,
                                 num_workers=4)

        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        score = ResScore(self.config).to(self.config.device)
        # score = ResEnergy(self.config).to(self.config.device)

        optimizer = self.get_optimizer(score.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

        step = 0

        sigma = self.config.training.noise_std

        for epoch in range(self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                step += 1

                X = X.to(self.config.device)  # [128, 3, 32, 32]
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)

                scaled_score = lambda x: score(x)

                if self.config.training.algo == 'ssm':
                    X = X + torch.randn_like(X) * sigma
                    loss, *_ = sliced_score_estimation_vr(scaled_score, X.detach(), n_particles=1)
                    # loss, *_ = sliced_score_matching_vr(scaled_score, X.detach(), n_particles=1)

                elif self.config.training.algo == 'dsm':
                    loss = dsm_score_estimation(scaled_score, X, sigma=self.config.training.noise_std)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_logger.add_scalar('loss', loss, global_step=step)
                tb_logger.add_scalar('sigma', sigma, global_step=step)
                # logging.info("step: {}, loss: {}, sigma: {}".format(step, loss.item(), sigma))
                print("step: {}, loss: {}, sigma: {}".format(step, loss.item(), sigma))

                if step >= self.config.training.n_iters:
                    return 0

                if step % 100 == 0:
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)
                    if self.config.data.logit_transform:
                        test_X = self.logit_transform(test_X)

                    if self.config.training.algo == 'ssm':
                        test_X += torch.randn_like(test_X) * self.config.training.noise_std
                        test_loss, *_ = sliced_score_estimation_vr(scaled_score, test_X.detach(), n_particles=1)
                    elif self.config.training.algo == 'dsm':
                        test_loss = dsm_score_estimation(scaled_score, test_X, sigma=self.config.training.noise_std)

                    tb_logger.add_scalar('test_loss', test_loss, global_step=step)
                    # logging.info("step: {}, test_loss: {}".format(step, test_loss.item()))
                    print("step: {}, test_loss: {}".format(step, test_loss.item()))

                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict()
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

                if step == self.config.training.n_iters:
                    return 0


class EnergyNetTrainer():
    def __init__(self,
                 energy_model,
                 energy_model_optim,
                 energy_model_lr,
                 energy_model_weight_decay,
                 energy_model_optim_beta1,
                 energy_model_noise_std,
                 energy_model_resume_training,
                 energy_model_algo="ssm",
                 energy_model_dir="checkpoints/fisher"):
        self.energy = energy_model
        self.optimizer = energy_model_optim
        self.lr = energy_model_lr
        self.weight_decay = energy_model_weight_decay
        self.beta1 = energy_model_optim_beta1
        self.noise_std = energy_model_noise_std

        self.resume_training = energy_model_resume_training
        self.algo = energy_model_algo
        self.model_dir = energy_model_dir

    def get_optimizer(self, parameters):
        if self.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.lr, weight_decay=self.weight_decay,
                              betas=(self.beta1, 0.999))
        elif self.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.optimizer))

    def launch(self,
               train_data,
               validate_data,
               logger,
               n_epochs,
               n_iters=None,
               snapshot_freq=None):

        test_loader = validate_data
        test_iter = iter(test_loader)

        # tb_path = os.path.join('run', 'tensorboard', '0')
        # if os.path.exists(tb_path):
        #     shutil.rmtree(tb_path)

        # tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)
        energy = self.energy
        optimizer = self.get_optimizer(energy.parameters())

        if self.resume_training:
            states = torch.load(os.path.join(self.model_dir, 'sm_checkpoint.pth'))
            energy.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])

            return

        flag = False
        step = 0

        sigma = self.noise_std

        for epoch in range(n_epochs):
            if flag: break
            for i, (X, y) in enumerate(train_data):
                step += 1

                # [B, D] has been to device
                scaled_score = lambda x: energy(x)

                if self.algo == 'ssm':
                    X = X + torch.randn_like(X) * sigma
                    loss, *_ = sliced_score_matching_vr(scaled_score, X.detach(), n_particles=1)

                elif self.algo == 'dsm':
                    loss = dsm(scaled_score, X, sigma=self.noise_std)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # tb_logger.add_scalar('loss', loss, global_step=step)
                # tb_logger.add_scalar('sigma', sigma, global_step=step)
                logger.logger.info("epoch: {}, step: {}, total_step: {}, loss: {}, sigma: {}".format(epoch, i, step, loss.item(), sigma))

                if step >= n_iters:
                    flag = True
                    break

                if step % 100 == 0:
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)

                    # test_X has been to device

                    if self.algo == 'ssm':
                        test_X += torch.randn_like(test_X) * self.noise_std
                        test_loss, *_ = sliced_score_matching_vr(scaled_score, test_X.detach(), n_particles=1)
                    elif self.algo == 'dsm':
                        test_loss = dsm(scaled_score, test_X, sigma=self.noise_std)

                    # tb_logger.add_scalar('test_loss', test_loss, global_step=step)
                    logger.logger.info("epoch: {}, step: {}, total_step: {}, [test_loss]: {}, sigma: {}".format(epoch, i, step, test_loss.item(), self.noise_std))

                if step % snapshot_freq == 0:
                    states = [
                        energy.state_dict(),
                        optimizer.state_dict()
                    ]
                    torch.save(states, os.path.join(self.model_dir, 'sm_checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.model_dir, 'sm_checkpoint.pth'))
        
        torch.save(states, os.path.join(self.model_dir, 'sm_checkpoint.pth'))
        return
        
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--run', type=str, default='run', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, default='0', help='A string for documentation purpose')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')

    args = parser.parse_args()
    args.log = os.path.join(args.run, 'logs', args.doc)

    config_namespace = dict2namespace(config)
    config_namespace.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    scorenet = ScoreNetRunner(args, config_namespace)
    scorenet.train()