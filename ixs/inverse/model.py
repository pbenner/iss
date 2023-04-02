## Copyright (C) 2023 Sofya Laskina, Philipp Benner
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.

## Imports
## ----------------------------------------------------------------------------

import dill
import torch
import pytorch_lightning as pl
import FrEIA.framework as Ff
import FrEIA.modules   as Fm

from copy import deepcopy
from sklearn.preprocessing import StandardScaler

from .          import losses
from .data      import ScatteringData
from .model_lit import LitProgressBar, LitMetricTracker, LitTensorDataset, LitModelWrapper

## ----------------------------------------------------------------------------

class InvertibleSASModelCore(torch.nn.Module):

    def __init__(
            self,
            # Data dimensions
            ndim_x               = None,
            ndim_y               = None,
            ndim_z               = None,
            ndim_pad_x           = None,
            ndim_pad_zy          = None,
            # Noise options
            add_y_noise          = 0,
            add_z_noise          = 2e-2,
            add_pad_noise        = 1e-2,
            y_uncertainty_sigma  = 0.12*4,
            # Other model options
            nblocks              = 5,
            hidden_layer_sizes   = 32,
            exponent_clamping    = 2,
            train_reconstruction = False,
            verbose_construction = False,
            ):

        super().__init__()

        self.ndim_x               = ndim_x
        self.ndim_y               = ndim_y
        self.ndim_z               = ndim_z
        self.ndim_pad_x           = ndim_pad_x
        self.ndim_pad_zy          = ndim_pad_zy

        self.add_y_noise          = add_y_noise
        self.add_z_noise          = add_z_noise
        self.add_pad_noise        = add_pad_noise
        self.y_uncertainty_sigma  = y_uncertainty_sigma

        self.lambd_fit_forw       = 0.01
        self.lambd_mmd_forw       = 100
        self.lambd_mmd_back       = 500
        self.lambd_reconstruct    = 1.0
        self.hidden_layer_sizes   = hidden_layer_sizes
        self.train_reconstruction = train_reconstruction

        self.mmd_forw_kernels     = [(0.2, 1/2), (1.5, 1/2), (3.0, 1/2)]
        self.mmd_back_kernels     = [(0.2, 1/2), (0.2, 1/2), (0.2, 1/2)]
        self.mmd_back_weighted    = True

        input = Ff.InputNode(ndim_x + ndim_pad_x, name='input')
        nodes = [input]

        for i in range(nblocks):
            nodes.append(Ff.Node( nodes[-1].out0 , Fm.RNVPCouplingBlock,
                                 { 'subnet_constructor': self.subnet, 'clamp': exponent_clamping }, name = f'coupling_{i}'))
            nodes.append(Ff.Node([nodes[-1].out0], Fm.PermuteRandom,
                                 {'seed' : i}, name = f'permute_{i}'))

        nodes.append(Ff.OutputNode([nodes[-1].out0], name='output'))

        self.freia_model = Ff.GraphINN(nodes, verbose = verbose_construction)

    def subnet(self, dims_in, dims_out):
        return torch.nn.Sequential(
            torch.nn.Linear(dims_in,                   self.hidden_layer_sizes*2), torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_sizes*2, self.hidden_layer_sizes  ), torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_sizes  , dims_out))

    def noise_batch(self, batch_size, ndim, device):
        """
        Adds Gaussian noise to the data
        """
        return torch.randn(batch_size, ndim, device=device)

    def _predict_forward(self, x):

        if self.ndim_pad_x:
            x = torch.cat((x, self.add_pad_noise * self.noise_batch(x.shape[0], self.ndim_pad_x, x.device)), dim=1)

        y, _ = self.freia_model(x, rev = False, jac = False)

        return y

    def _predict_backward(self, y):

        if self.add_y_noise > 0:
            y += self.add_y_noise * self.noise_batch(y.shape[0], self.ndim_y, y.device)
        if self.ndim_pad_zy > 0:
            y = torch.cat((self.add_pad_noise * self.noise_batch(y.shape[0], self.ndim_pad_zy, y.device), y), dim=1)

        yz = torch.cat((self.noise_batch(y.shape[0], self.ndim_z, y.device), y), dim=1)

        x, _ = self.freia_model(yz, rev = True, jac = False)

        return x

    def forward(self, x_or_y, rev = False):

        if rev:
            return self._predict_backward(x_or_y)
        else:
            return self._predict_forward (x_or_y)

    def loss_backward_mmd(self, x, y):
        """
        Calculates the MMD loss in the backward direction
        """
        x_samples, _ = self.freia_model(y, rev = True, jac = False) 

        MMD = losses.backward_mmd(x, x_samples, self.mmd_back_kernels) 

        if self.mmd_back_weighted:
            MMD *= torch.exp(- 0.5 / self.y_uncertainty_sigma**2 * losses.l2_dist_matrix(y, y))

        return self.lambd_mmd_back * torch.mean(MMD)

    def loss_forward_mmd(self, out, y):
        """
        Calculate MMD loss in the forward direction
        """
        output_block_grad = torch.cat(
            (out[:, :self.ndim_z ],
             out[:, -self.ndim_y:].data), dim=1) 
        y_short = torch.cat((y[:, :self.ndim_z], y[:, -self.ndim_y:]), dim=1)

        l_forw_fit = self.lambd_fit_forw * losses.l2_fit(out[:, self.ndim_z:], y[:, self.ndim_z:])
        l_forw_mmd = self.lambd_mmd_forw * torch.mean(losses.forward_mmd(output_block_grad, y_short, self.mmd_forw_kernels))

        return l_forw_fit, l_forw_mmd

    def loss_reconstruction(self, out_y, x):
        """
        Calculate reconstruction loss
        """
        cat_inputs = [ out_y[:, :self.ndim_z] + self.add_z_noise * self.noise_batch(out_y.shape[0], self.ndim_z, out_y.device) ]
        
        if self.ndim_pad_zy:
            cat_inputs.append(out_y[:, self.ndim_z:-self.ndim_y] + self.add_pad_noise * self.noise_batch(out_y.shape[0], self.ndim_pad_zy, out_y.device)) # list with 2 tensor

        cat_inputs.append(out_y[:, -self.ndim_y:] + self.add_y_noise * self.noise_batch(out_y.shape[0], self.ndim_y, out_y.device))
        x_reconstructed, _ = self.freia_model(torch.cat(cat_inputs, 1), rev = True, jac = False)

        return self.lambd_reconstruct * losses.l2_fit(x_reconstructed[:, :self.ndim_pad_x], x[:,:self.ndim_pad_x], self.batch_size)

    def loss(self, x, y):

        if self.ndim_pad_x > 0:
            x = torch.cat((x, self.add_pad_noise * self.noise_batch(x.shape[0], self.ndim_pad_x, x.device)), dim=1) 
        if self.add_y_noise > 0:
            y += self.add_y_noise * self.noise_batch(y.shape[0], self.ndim_y, y.device)
        if self.ndim_pad_zy > 0:
            y = torch.cat((self.add_pad_noise * self.noise_batch(y.shape[0], self.ndim_pad_zy, y.device), y), dim=1)

        y = torch.cat((self.noise_batch(y.shape[0], self.ndim_z, y.device), y), dim=1)

        y_hat, _ = self.freia_model(x, jac = False)

        # Evaluate all three losses
        loss_forward  = self.loss_forward_mmd(y_hat, y)
        loss_backward = self.loss_backward_mmd(x, y)
        loss_reconst  = 0

        if self.train_reconstruction:
            loss_reconst = self.loss_reconstruction(y_hat.data, x)

        # Aggregate losses
        loss_sum = sum(loss_forward) + loss_backward + loss_reconst
        # Save loss components for logging
        loss_components = {
            'forw_fit': loss_forward[0],
            'forw_mmd': loss_forward[1],
            'back_fit': loss_backward
        }
        if self.train_reconstruction:
            loss_components['reconst'] = loss_reconst

        return y_hat, loss_sum, loss_components

## ----------------------------------------------------------------------------

class InvertibleSASModel():

    def __init__(self,
            *args,
            # Model options
            **kwargs):

        self.lit_model = LitModelWrapper(InvertibleSASModelCore, *args, **kwargs)
        self.scaler    = StandardScaler()

    def cross_validation(self, data : ScatteringData, n_splits, shuffle = True, random_state = 42):

        if not isinstance(data, ScatteringData):
            raise ValueError(f'Data must be given as CoordinationFeaturesData, but got type {type(data)}')

        return self.lit_model._cross_validation(data, n_splits, shuffle, random_state)

    def train(self, data : ScatteringData):

        data.fit_scaler(self.scaler)
        data.normalize(self.scaler)

        return self.lit_model._train(data)

    def predict_forward(self, x : torch.Tensor):

        return self.lit_model.model(x, rev = False)

    def predict_backward(self, y : torch.Tensor):

        return self.lit_model.model(y, rev = True)

    @classmethod
    def load(cls, filename : str) -> 'InvertibleSASModel':

        with open(filename, 'rb') as f:
            model = dill.load(f)

        if not isinstance(model, cls):
            raise ValueError(f'file {filename} contains incorrect model class {type(model)}')

        return model

    def save(self, filename : str) -> None:

        with open(filename, 'wb') as f:
            dill.dump(self, f)
