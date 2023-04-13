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
import FrEIA.framework as Ff
import FrEIA.modules   as Fm

from copy                    import deepcopy
from sklearn.model_selection import KFold

from .          import losses
from .data      import ScatteringData
from .model_lit import LitModelWrapper
from .scaler    import SASScaler

## ----------------------------------------------------------------------------

class InvertibleSASModelCore(torch.nn.Module):

    def __init__(
            self,
            metadata             = None,
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

        if metadata is None:
            raise ValueError('metadata is a required keyword argument')

        self.ndim_x               = metadata['ndim_x']
        self.ndim_y               = metadata['ndim_y']
        self.ndim_z               = metadata['ndim_z']
        self.ndim_pad_x           = metadata['ndim_pad_x']
        self.ndim_pad_zy          = metadata['ndim_pad_zy']

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

        self.scaler               = SASScaler(self.ndim_x, self.ndim_y, metadata['shapes_dict'])

        input = Ff.InputNode(self.ndim_x + self.ndim_pad_x, name='input')
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

        assert len(x.shape) == 2, 'Input data has invalid dimension'
        assert x.shape[1] == self.ndim_x, 'Input data has invalid dimension'

        if self.ndim_pad_x:
            x = torch.cat((x, self.add_pad_noise * self.noise_batch(x.shape[0], self.ndim_pad_x, x.device)), dim=1)

        x_norm   = self.scaler.normalize_input(x)
        y_hat, _ = self.freia_model(x_norm, rev = False, jac = False)
        y_hat    = self.scaler.denormalize_output(y_hat)

        return y_hat, x

    def _predict_backward(self, y):

        assert len(y.shape) == 2, 'Input data has invalid dimension'
        assert y.shape[1] == self.ndim_y, 'Input data has invalid dimension'

        if self.add_y_noise > 0:
            y += self.add_y_noise * self.noise_batch(y.shape[0], self.ndim_y, y.device)
        if self.ndim_pad_zy > 0:
            y = torch.cat((self.add_pad_noise * self.noise_batch(y.shape[0], self.ndim_pad_zy, y.device), y), dim=1)

        y = torch.cat((self.noise_batch(y.shape[0], self.ndim_z, y.device), y), dim=1)

        y_norm   = self.scaler.normalize_output(y)
        x_hat, _ = self.freia_model(y_norm, rev = True, jac = False)
        x_hat    = self.scaler.denormalize_input(x_hat)

        return x_hat, y

    def forward(self, x_or_y, rev = False):

        if rev:
            x_hat, _ = self._predict_backward(x_or_y)
            # Truncate padded part
            x_hat = x_hat[:, 0:self.ndim_x]
            return x_hat
        else:
            y_hat, _ = self._predict_forward (x_or_y)
            # Truncate padded part
            y_hat = y_hat[:, 0:self.ndim_y]
            return y_hat

    def loss_mmd_backward(self, x, x_hat, y):
        """
        Calculates the MMD loss in the backward direction
        """
        MMD = losses.backward_mmd(x, x_hat, self.mmd_back_kernels)

        if self.mmd_back_weighted:
            MMD *= torch.exp(- 0.5 / self.y_uncertainty_sigma**2 * losses.l2_dist_matrix(y, y))

        return self.lambd_mmd_back * torch.mean(MMD)

    def loss_mmd_forward(self, y, y_hat):
        """
        Calculate MMD loss in the forward direction
        """
        output_block_grad = torch.cat(
            (y_hat[:, :self.ndim_z ],
             y_hat[:, -self.ndim_y:].data), dim=1)
        y_short = torch.cat((y[:, :self.ndim_z], y[:, -self.ndim_y:]), dim=1)

        l_forw_fit = self.lambd_fit_forw * losses.l2_fit(y_hat[:, self.ndim_z:], y[:, self.ndim_z:])
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

        # Augment x/y and predict in both directions. The augmented
        # tensors are required to compute the loss
        y_hat, x = self._predict_forward (x)
        x_hat, y = self._predict_backward(y)

        # Evaluate all three losses
        loss_forward  = self.loss_mmd_forward (y, y_hat)
        loss_backward = self.loss_mmd_backward(x, x_hat, y)
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

        return x_hat, y_hat, loss_sum, loss_components

## ----------------------------------------------------------------------------

class InvertibleSASModel():

    def __init__(self, **kwargs):

        self.lit_model = LitModelWrapper(InvertibleSASModelCore, **kwargs)

    def cross_validation(self, data : ScatteringData, n_splits, shuffle = True, random_state = 42):

        if not isinstance(data, ScatteringData):
            raise ValueError(f'Data must be given as CoordinationFeaturesData, but got type {type(data)}')

        return self.lit_model._cross_validation(data, n_splits, shuffle, random_state)

    def train(self, data : ScatteringData):

        # We need the whole data for fitting the scalers, not just batches
        self.lit_model.model.scaler.fit_input (data.X)
        self.lit_model.model.scaler.fit_output(data.y)

        best_model, stats = self.lit_model._train(data)

        self.lit_model = best_model

        return stats

    def test(self, data : ScatteringData):

        return self.lit_model._test(data)

    def predict_forward(self, data : ScatteringData):

        with torch.no_grad():
            y = self.lit_model.model(data.X, rev = False)

        return y

    def predict_backward(self, data : ScatteringData):

        with torch.no_grad():
            X = self.lit_model.model(data.y, rev = True)

        return X

    def cross_validation(self, data : ScatteringData, n_splits, shuffle = True, random_state = 42):

        if n_splits < 2:
            raise ValueError(f'k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits={n_splits}')

        x_hat  = torch.tensor([], dtype = torch.float)
        x      = torch.tensor([], dtype = torch.float)
        y_hat  = torch.tensor([], dtype = torch.float)
        y      = torch.tensor([], dtype = torch.float)
        losses = []

        initial_model = self.lit_model

        for fold, (index_train, index_test) in enumerate(KFold(n_splits, shuffle = shuffle, random_state = random_state).split(data)):

            print(f'Training fold {fold+1}/{n_splits}...')

            data_train = data.subset(index_train)
            data_test  = data.subset(index_test )

            # Clone model
            self.lit_model = deepcopy(initial_model)

            # Train model
            best_val_score = self.train(data_train)['best_val_error']

            # Test model
            test_x, test_x_hat, test_y, test_y_hat, stats = self.test(data_test)

            # Print score
            print(f'Best validation score: {best_val_score}')

            # Save predictions for model evaluation
            x_hat  = torch.cat((x_hat, test_x_hat))
            x      = torch.cat((x    , test_x    ))
            y_hat  = torch.cat((y_hat, test_y_hat))
            y      = torch.cat((y    , test_y    ))
            # Save loss on test data
            losses.append(stats['test_loss'])

        # Reset model
        self.lit_model = initial_model

        # Compute average loss
        test_loss = torch.tensor(losses).mean().item()

        return x, x_hat, y, y_hat, test_loss

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
