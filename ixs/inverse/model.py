## Copyright (C) 2023 Sofya Laskina, Philipp Benner
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## Imports
## ----------------------------------------------------------------------------

import dill
import torch
import pytorch_lightning as pl

import FrEIA.framework as Ff
import FrEIA.modules   as Fm

from copy import deepcopy

from .          import losses
from .data      import ScatteringData
from .model_lit import LitProgressBar, LitMetricTracker, LitTensorDataset, LitModelWrapper

## ----------------------------------------------------------------------------

class InvertibleSASModelCore(torch.nn.Module):

    def __init__(
            self,
            # Data dimensions
            ndim_x,
            ndim_y,
            ndim_z,
            ndim_pad_x,
            ndim_pad_zy,
            # Noise options
            add_y_noise          = 0,
            add_z_noise          = 2e-2,
            add_pad_noise        = 1e-2,
            y_uncertainty_sigma  = 0.12*4,
            # Other model options
            nblocks              = 5,
            hidden_layer_sizes   = 32,
            exponent_clamping    = 2,
            verbose_construction = False,
            ):

        self.ndim_x              = ndim_x
        self.ndim_y              = ndim_y
        self.ndim_z              = ndim_z
        self.ndim_pad_x          = ndim_pad_x
        self.ndim_pad_zy         = ndim_pad_zy

        self.add_y_noise         = add_y_noise
        self.add_z_noise         = add_z_noise
        self.add_pad_noise       = add_pad_noise
        self.y_uncertainty_sigma = y_uncertainty_sigma

        self.lambd_fit_forw      = 0.01 
        self.lambd_mmd_forw      = 100
        self.lambd_mmd_back      = 500
        self.lambd_reconstruct   = 1.0
        self.hidden_layer_sizes  = hidden_layer_sizes

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

    def forward(self, x, **kwargs):

        if self.ndim_pad_x:
            x = torch.cat((x, self.add_pad_noise * self.noise_batch(self.ndim_pad_x)), dim=1)

        self.freia_model(x, **kwargs)

    def loss_backward_mmd(self, x, y):
        """
        Calculates the MMD loss in the backward direction
        """
        x_samples, _ = self.model(y, rev = True, jac = False) 

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

        l_forw_fit = self.lambd_fit_forw * losses.l2_fit(out[:, self.ndim_z:], y[:, self.ndim_z:], self.batch_size)
        l_forw_mmd = self.lambd_mmd_forw * torch.mean(losses.forward_mmd(output_block_grad, y_short, self.mmd_forw_kernels))

        return l_forw_fit, l_forw_mmd

    def loss_reconstruction(self, out_y, x):
        """
        Calculate reconstruction loss
        """
        cat_inputs = [ out_y[:, :self.ndim_z] + self.add_z_noise * self.noise_batch(self.ndim_z) ]
        
        if self.ndim_pad_zy:
            cat_inputs.append(out_y[:, self.ndim_z:-self.ndim_y] + self.add_pad_noise * self.noise_batch(self.ndim_pad_zy)) # list with 2 tensor

        cat_inputs.append(out_y[:, -self.ndim_y:] + self.add_y_noise * self.noise_batch(self.ndim_y))
        x_reconstructed, _ = self.freia_model(torch.cat(cat_inputs, 1), rev = True, jac = False)

        return self.lambd_reconstruct * losses.l2_fit(x_reconstructed[:, :self.ndim_pad_x], x[:,:self.ndim_pad_x], self.batch_size)

    def loss(self, x, y):

        if self.add_y_noise > 0:
            y += self.add_y_noise * self.noise_batch(self.ndim_y)
        if self.ndim_pad_x:
            x = torch.cat((x, self.add_pad_noise * self.noise_batch(self.ndim_pad_x)), dim=1) 
        if self.ndim_pad_zy:
            y = torch.cat((self.add_pad_noise * self.noise_batch(self.ndim_pad_zy), y), dim=1)

        y = torch.cat((self.noise_batch(self.ndim_z), y), dim=1)

        y_hat, _ = self.freia_model(x, jac = False)

        r = self.loss_forward_mmd(y_hat, y)
        r.append(self.loss_backward_mmd(x, y))

        if self.train_reconstruction:
            r.append(self.loss_reconstruction(y_hat.data, x))

        return torch.sum(r)

## ----------------------------------------------------------------------------

class InvertibleSASModel():

    def __init__(self,
            # Trainer options
            patience = 100, max_epochs = 1000, accelerator = 'gpu', devices = [0], strategy = None,
            # Data options
            val_size = 0.1, batch_size = 128, num_workers = 2,
            # Model options
            **kwargs):

        self.lit_model           = LitModelWrapper(InvertibleSASModelCore, **kwargs)
        self.lit_trainer         = None
        self.lit_trainer_options = {
            'patience'    : patience,
            'max_epochs'  : max_epochs,
            'accelerator' : accelerator,
            'devices'     : devices,
            'strategy'    : strategy,
        }
        self.lit_data_options    = {
            'val_size'    : val_size,
            'batch_size'  : batch_size,
            'num_workers' : num_workers,
        }

    def _setup_trainer_(self):
        self.lit_matric_tracker      = LitMetricTracker()
        self.lit_early_stopping      = pl.callbacks.EarlyStopping(monitor = 'val_loss', patience = self.lit_trainer_options['patience'])
        self.lit_checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k = 1, monitor = 'val_loss', mode = 'min')

        self.lit_trainer = pl.Trainer(
            enable_checkpointing = True,
            enable_progress_bar  = True,
            logger               = False,
            max_epochs           = self.lit_trainer_options['max_epochs'],
            accelerator          = self.lit_trainer_options['accelerator'],
            devices              = self.lit_trainer_options['devices'],
            strategy             = self.lit_trainer_options['strategy'],
            callbacks            = [LitProgressBar(), self.lit_early_stopping, self.lit_checkpoint_callback, self.lit_matric_tracker])

    def cross_validation(self, data : ScatteringData, n_splits, shuffle = True, random_state = 42):

        if not isinstance(data, ScatteringData):
            raise ValueError(f'Data must be given as CoordinationFeaturesData, but got type {type(data)}')

        if n_splits < 2:
            raise ValueError(f'k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits={n_splits}')

        data  = LitTensorDataset(data, n_splits = n_splits, shuffle = shuffle, random_state = random_state, **self.lit_data_options)

        return self._cross_validation(data)

    def _cross_validation(self, data : LitTensorDataset):

        y_hat = torch.tensor([], dtype = torch.float)
        y     = torch.tensor([], dtype = torch.float)

        initial_model = self.lit_model.model

        for fold in range(data.n_splits):

            print(f'Training fold {fold+1}/{data.n_splits}...')
            data.setup_fold(fold)

            # Clone model
            self.lit_model.model = deepcopy(initial_model)

            # Train model
            best_val_score = self._train(data)['best_val_error']

            # Test model
            self.lit_trainer.test(self.lit_model, data)
            test_y, test_y_hat = self.lit_model.test_y, self.lit_model.test_y_hat

            # Print score
            print(f'Best validation score: {best_val_score}')

            # Save predictions for model evaluation
            y_hat = torch.cat((y_hat, test_y_hat))
            y     = torch.cat((y    , test_y    ))

        # Compute final test score
        test_loss = self.lit_model.loss(y_hat, y).item()

        return test_loss, y, y_hat

    def train(self, data : ScatteringData):

        data = LitTensorDataset(data, self.lit_model.model.model_config, **self.lit_data_options)

        return self._train(data)

    def _train(self, data : LitTensorDataset):

        # We always need a new trainer for training the model
        self._setup_trainer_()

        # Train model on train data and use validation data for early stopping
        self.lit_trainer.fit(self.lit_model, data)

        # Get best model
        self.lit_model = self.lit_model.load_from_checkpoint(self.lit_checkpoint_callback.best_model_path)

        result = {
            'best_val_error': self.lit_checkpoint_callback.best_model_score.item(),
            'train_error'   : torch.stack(self.lit_matric_tracker.train_error).tolist(),
            'val_error'     : torch.stack(self.lit_matric_tracker.val_error  ).tolist() }

        return result

    def predict(self, data : ScatteringData):

        data = LitTensorDataset(data, self.lit_model.model.model_config, **self.lit_data_options)

        return self._predict(data)

    def _predict(self, data : LitTensorDataset):

        if self.lit_trainer is None:
            self._setup_trainer_()

        y_hat_batched = self.lit_trainer.predict(self.lit_model, data)

        return torch.cat(y_hat_batched, dim=0)

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
