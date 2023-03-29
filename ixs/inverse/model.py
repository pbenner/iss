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

class InvertibleSASModelCore():

    def __init__(
            self,
            ndim_x,
            ndim_pad_x,
            init_scale = 0.10,
            N_blocks   = 5,
            hidden_layer_sizes = 32,
            exponent_clamping = 2,
            verbose_construction = False,
            ):

        self.hidden_layer_sizes = hidden_layer_sizes

        input = Ff.InputNode(ndim_x + ndim_pad_x, name='input')
        nodes = [input]

        for i in range(N_blocks):
            nodes.append(Ff.Node( nodes[-1].out0 , Fm.RNVPCouplingBlock, { 'subnet_constructor': self.subnet, 'clamp': exponent_clamping }, name = f'coupling_{i}'))
            nodes.append(Ff.Node([nodes[-1].out0], Fm.PermuteRandom, {'seed':i}, name=f'permute_{i}'))

        nodes.append(Ff.OutputNode([nodes[-1].out0], name='output'))
        model = Ff.GraphINN(nodes, verbose = verbose_construction)

    def subnet(self, dims_in, dims_out):
        return torch.nn.Sequential(
                    torch.nn.Linear(dims_in, self.hidden_layer_sizes*2), torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_layer_sizes*2,  self.hidden_layer_sizes), torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_layer_sizes,  dims_out))

## ----------------------------------------------------------------------------

class InvertibleSASModel():

    def __init__(self,
            # Trainer options
            patience = 100, max_epochs = 1000, accelerator = 'gpu', devices = [0], strategy = None,
            # Data options
            val_size = 0.1, batch_size = 128, num_workers = 2,
            # Model options
            **kwargs):

        self.lit_model           = LitModelWrapper(**kwargs)
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

## ----------------------------------------------------------------------------

class ScatteringModel():
    """ a class to wrap up a scattering problem learning suite with user-defined hyperparameters"""
    def __init__(self, filename_out, 
                        device, 
                        lr_init, 
                        batch_size, 
                        n_epochs, 
                        n_its_per_epoch, 
                        pre_low_lr, 
                        final_decay, 
                        l2_weight_reg, 
                        adam_betas, 
                        ndim_pad_x, 
                        ndim_y, 
                        ndim_z, 
                        ndim_pad_zy , 
                        train_reconstruction, 
                        lambd_fit_forw , 
                        lambd_mmd_forw, 
                        lambd_mmd_back, 
                        lambd_reconstruct, 
                        add_y_noise , 
                        add_z_noise, 
                        add_pad_noise, 
                        y_uncertainty_sigma,
                        mmd_forw_kernels, 
                        mmd_back_kernels, 
                        mmd_back_weighted, 
                        init_scale, 
                        flow):
    
        self.filename_out          = filename_out
        self.device                = device
        self.lr_init               = lr_init
        self.batch_size            = batch_size
        self.n_epochs              = n_epochs
        self.n_its_per_epoch = n_its_per_epoch
        self.pre_low_lr            = pre_low_lr
        self.final_decay           = final_decay
        self.l2_weight_reg         = l2_weight_reg
        self.adam_betas            = adam_betas
        self.ndim_pad_x            = ndim_pad_x
        self.ndim_y                = ndim_y
        self.ndim_z                = ndim_z
        self.ndim_pad_zy           = ndim_pad_zy 
        self.train_reconstruction  = train_reconstruction
        self.lambd_fit_forw        = lambd_fit_forw 
        self.lambd_mmd_forw        = lambd_mmd_forw
        self.lambd_mmd_back        = lambd_mmd_back
        self.lambd_reconstruct     = lambd_reconstruct
        self.add_y_noise           = add_y_noise 
        self.add_z_noise           = add_z_noise
        self.add_pad_noise         = add_pad_noise
        self.y_uncertainty_sigma   = y_uncertainty_sigma
        self.mmd_forw_kernels      = mmd_forw_kernels
        self.mmd_back_kernels      = mmd_back_kernels
        self.mmd_back_weighted     = mmd_back_weighted
        self.init_scale            = init_scale
        self.flow                  = flow

    def set_model(self, model):
        """
        Set the design of NN
        Arguments:
            model: torch model
        """
        self.model = model
        self.model.to(self.device)

    def set_optimizer(self):
        """
        set on optimizer with hyperparmaeters defined as initializing the class or after calling the `update_hyperparameters' function
        """
        self.params_trainable = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        for p in self.params_trainable:
            p.data = self.init_scale * torch.randn(p.data.shape).to(self.device)

        gamma = (self.final_decay)**(1./self.n_epochs)
        self.optim = torch.optim.Adam(self.params_trainable, lr=self.lr_init, betas=self.adam_betas, eps=1e-6, weight_decay=self.l2_weight_reg)
        self.weight_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=gamma)

    def loss_backward_mmd(self,x, y):
        """
        Calculates the MMD loss in the backward direction
        """
        x_samples, x_samples_jac = self.model(y, rev=True, jac = True) 
        MMD = losses.backward_mmd(x, x_samples, self.mmd_back_kernels, self.device) 
        if self.mmd_back_weighted:
            MMD *= torch.exp(- 0.5 / self.y_uncertainty_sigma**2 * losses.l2_dist_matrix(y, y))
        return self.lambd_mmd_back * torch.mean(MMD)

    def loss_forward_mmd(self, out, y):
        """
        Calculates  MMD loss in the forward direction
        """
        output_block_grad = torch.cat((out[:, :self.ndim_z],
                                    out[:, -self.ndim_y:].data), dim=1) 
        y_short = torch.cat((y[:, :self.ndim_z], y[:, -self.ndim_y:]), dim=1)

        l_forw_fit = self.lambd_fit_forw * losses.l2_fit(out[:, self.ndim_z:], y[:, self.ndim_z:], self.batch_size)
        l_forw_mmd = self.lambd_mmd_forw  * torch.mean(losses.forward_mmd(output_block_grad, y_short, self.mmd_forw_kernels, self.device))

        return l_forw_fit, l_forw_mmd

    def loss_reconstruction(self, out_y, x):
        """
        calculates reconstruction loss
        """
        cat_inputs = [out_y[:, :self.ndim_z] + self.add_z_noise * self.noise_batch(self.ndim_z)] # list with 1 tensor
        
        if self.ndim_pad_zy:
            cat_inputs.append(out_y[:, self.ndim_z:-self.ndim_y] + self.add_pad_noise * self.noise_batch(self.ndim_pad_zy)) # list with 2 tensor
        cat_inputs.append(out_y[:, -self.ndim_y:] + self.add_y_noise * self.noise_batch(self.ndim_y)) # list with 3 tensors
        x_reconstructed, x_reconstructed_jac = self.model(torch.cat(cat_inputs, 1), rev=True, jac = True) # concatenate list elements along axis 1
        return self.lambd_reconstruct * losses.l2_fit(x_reconstructed[:, :self.ndim_pad_x], x[:,:self.ndim_pad_x], self.batch_size)

    def noise_batch(self,ndim):
        """
        adds normlly distributed noize to the data
        """
        return torch.randn(self.batch_size, ndim).to(self.device)

    def train_epoch(self, test=False):
        """
        the training loop over one epoch
        """
        if not test:
            self.model.train()
            loader = self.train_loader

        else:
            self.model.eval()
            loader = self.test_loader
            nograd = torch.no_grad()
            nograd.__enter__()

        batch_idx = 0
        loss_history = []

        for x, y in loader:

            if batch_idx > self.n_its_per_epoch:
                break
            batch_losses = []

            batch_idx += 1

            x, y = x.to(self.device), y.to(self.device)

            if self.add_y_noise > 0:
                y += self.add_y_noise * self.noise_batch(self.ndim_y)
            if self.ndim_pad_x:
                x = torch.cat((x, self.add_pad_noise * self.noise_batch(self.ndim_pad_x)), dim=1)
        
            if self.ndim_pad_zy:
                y = torch.cat((self.add_pad_noise * self.noise_batch(self.ndim_pad_zy), y), dim=1)
            y = torch.cat((self.noise_batch(self.ndim_z), y), dim=1)

            out_y, out_y_jac = self.model(x, jac        = True)

            batch_losses.extend(self.loss_forward_mmd(out_y, y))
            batch_losses.append(self.loss_backward_mmd(x, y))
            if self.train_reconstruction:
                batch_losses.append(self.loss_reconstruction(out_y.data, x))

            l_total = sum(batch_losses)
            loss_history.append([l.item() for l in batch_losses]) # lisr of lists: list for each batch

            if not test:
                l_total.backward()
                self.optim_step()

        if test:
            monitoring.show_hist(out_y[:, :self.ndim_z])
            monitoring.show_cov(out_y[:, :self.ndim_z])
            nograd.__exit__(None, None, None)
        return np.mean(loss_history, axis=0)

    def train(self):
        if self.flow == 'inverse':
            monitoring.restart()

        try:
            t_start = time()
            for i_epoch in range(-self.pre_low_lr, self.n_epochs):

                if i_epoch < 0:
                    for param_group in self.optim.param_groups:
                        param_group['lr'] = self.lr_init * 1e-1

                train_losses = self.train_epoch() # mean over batches
                test_losses        = self.train_epoch(test=True)
                t = np.concatenate([train_losses, test_losses])
                if self.flow == 'inverse':
                    monitoring.show_loss(t)
                else:
                    print('Epoch {i_e}: training loss: {tl}, test loss: {testl}'.format(i_e = i_epoch, tl = t[0], testl = t[1]))
        except:
            self.save(self.filename_out + '_ABORT')
            raise

        finally:
            print("\n\nTraining took %f minutes\n\n" % ((time()-t_start)/60.))
            self.save(self.filename_out)

    def scale_back(self, predictions):
        """
        only applicable for the inverse flow. Scale back the predicted values with the respective mean and standard deviation of initial input data. 
        Arguments:
            predictions (torch.Tensor): predicted values of the scatterer
        Returns:
            torch.Tensor: rescaled predictions
        """
        scaled_predictions = np.copy(predictions)
        for i in range(7):
            x = predictions[:, len(self.shapes_dict.keys())+i]
            
            scaled_predictions[:,len(self.shapes_dict.keys())+i] = x * float(self.scaler[i, 1]) + float(self.scaler[i,0])
        return scaled_predictions

    def predict(self, data, shapes):
        """
        make prediction on some data
        """

        if self.flow == 'inverse':
            y = torch.cat((self.add_pad_noise * torch.randn( len(data), self.ndim_pad_zy).to(self.device), data.to(self.device)), dim=1)
            y = torch.cat((torch.randn( len(data), self.ndim_z).to(self.device), y), dim=1)
            pred, _ = self.model(y, rev = True) 
            predictions =  pred.cpu().detach()
            return np.concatenate((predictions[:,:shapes],self.scaler.inverse_transform(predictions[:,shapes:self.ndim_x])), axis=1)
        else:
            self.model.to('cpu')
            try:
                return self.model(data).cpu().detach()
            except AttributeError:
                return self.model(data).cpu().detach()

    def make_prediction_test(self, data_subset):
        """
        run trained Neural network to predict the scatterer parameters
        """
        if self.flow == 'inverse':
            y = torch.cat((self.add_pad_noise * torch.randn( len(self.labels[data_subset]), self.ndim_pad_zy).to(self.device), self.labels[data_subset].to(self.device)), dim=1)
            y = torch.cat((torch.randn( len(self.labels[data_subset]), self.ndim_z).to(self.device), y), dim=1)
            pred, _ = self.model(y, rev = True) 
            predictions =  pred.cpu().detach()
            return np.concatenate((predictions[:,:len(self.shapes_dict.keys())],self.scaler.inverse_transform(predictions[:,len(self.shapes_dict.keys()):self.ndim_x])), axis=1)
        else:
            self.model.to('cpu')
            try:
                return self.model(self.inputs_norm[data_subset]).cpu().detach()
            except AttributeError:
                return self.model(self.inputs[data_subset]).cpu().detach()

    def create_table_from_outcomes_test(self, pred, data_subset):
        """
        creates a table with all predicted vs true values
        """
        cols= [['true_'+i, 'pred_'+i] for i in self.input_features]
        if self.flow == 'inverse':
            sampled_inputs = self.inputs[data_subset]
        else:
            sampled_inputs = self.labels[data_subset]
        df = pd.DataFrame(columns =  [i for c in cols for i in c  ], index = [])
        shapes = len(self.shapes_dict.keys())
        df['true_shape'] = sampled_inputs[:,:shapes].argmax(axis=1)
        df['pred_shape'] = pred[:,:shapes].argmax(axis=1)
        df['true_radius'] = np.take_along_axis(sampled_inputs[:,shapes:2*shapes],df.true_shape.values.reshape(-1,1), axis=1)
        df['pred_radius'] = np.take_along_axis(pred[:,shapes:2*shapes],df.pred_shape.values.reshape(-1,1), axis=1)
        for i,c in enumerate(df.columns[4:]):
            if 'pred' in c:
                df[c] = pred[:,i//2+2*shapes]
            elif 'true' in c:
                df[c] = sampled_inputs[:,i//2+2*shapes]
        return df

    def create_table_from_outcomes(self, pred, data):
        """
        creates a table with all predicted vs true values
        """
        cols= [['true_'+i, 'pred_'+i] for i in self.input_features]
        sampled_inputs = data
        df = pd.DataFrame(columns =  [i for c in cols for i in c  ], index = [])
        shapes = len(self.shapes_dict.keys())
        df['true_shape'] = sampled_inputs[:,:shapes].argmax(axis=1)
        df['pred_shape'] = pred[:,:shapes].argmax(axis=1)
        df['true_radius'] = np.take_along_axis(sampled_inputs[:,shapes:2*shapes],df.true_shape.values.reshape(-1,1), axis=1)
        df['pred_radius'] = np.take_along_axis(pred[:,shapes:2*shapes],df.pred_shape.values.reshape(-1,1), axis=1)
        for i,c in enumerate(df.columns[4:]):
            if 'pred' in c:
                df[c] = pred[:,i//2+2*shapes]
            elif 'true' in c:
                df[c] = sampled_inputs[:,i//2+2*shapes]
        return df

    def predict_latent(self, data_subset):
        """
        Look at the latent space variables calculated for the data
        Arguments:
            data_subset(list): indices subsetting data
        """
        x = self.inputs_norm[data_subset].to(self.device)
        x = torch.cat((x, self.add_pad_noise * torch.randn(1500, self.ndim_pad_x).to(self.device)), dim=1)
        out_y, _ = self.model(x, jac  = True)
        return out_y[:,:2].cpu().detach()
