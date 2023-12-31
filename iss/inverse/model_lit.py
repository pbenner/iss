## Copyright (C) 2023 Philipp Benner
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

import torch
import pytorch_lightning as pl
import os
import shutil
import sys

from tqdm   import tqdm
from typing import Optional

## ----------------------------------------------------------------------------

import logging
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

## ----------------------------------------------------------------------------

class LitMetricTracker(pl.callbacks.Callback):
    def __init__(self):
        self.val_error_batch   = []
        self.val_error         = []
        self.train_error_batch = []
        self.train_error       = []
        self.test_x            = []
        self.test_x_hat        = []
        self.test_y            = []
        self.test_y_hat        = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.train_error_batch.append(outputs['loss'].item())

    def on_train_epoch_end(self, *args, **kwargs):
        self.train_error.append(torch.mean(torch.tensor(self.train_error_batch)).item())
        self.train_error_batch = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.val_error_batch.append(outputs['val_loss'].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_error.append(torch.mean(torch.tensor(self.val_error_batch)).item())
        self.val_error_batch = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.test_x    .append(outputs['x'    ].detach().cpu())
        self.test_x_hat.append(outputs['x_hat'].detach().cpu())
        self.test_y    .append(outputs['y'    ].detach().cpu())
        self.test_y_hat.append(outputs['y_hat'].detach().cpu())

    @property
    def test_predictions(self):
        x     = torch.cat(self.test_x)
        x_hat = torch.cat(self.test_x_hat)
        y     = torch.cat(self.test_y)
        y_hat = torch.cat(self.test_y_hat)
        return x, x_hat, y, y_hat

## ----------------------------------------------------------------------------

class LitProgressBar(pl.callbacks.progress.TQDMProgressBar):
    def init_train_tqdm(self):
            """Override this to customize the tqdm bar for training."""
            bar = tqdm(
                desc=self.train_description,
                position=(2 * self.process_position),
                disable=self.is_disabled,
                leave=True,
                dynamic_ncols=True,
                bar_format='{desc}{percentage:3.0f}%|{postfix}',
                file=sys.stdout,
                smoothing=0,
            )
            return bar
    # Disable validation progress bar
    def on_validation_start(self, trainer, pl_module):
        pass
    def on_validation_end(self, trainer, pl_module):
        pass

## ----------------------------------------------------------------------------

class LitVerboseOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizer):
        self.optimizer         = optimizer
        self.param_groups_copy = None
        self._copy_parameters()

    def _copy_parameters(self):
        self.param_groups_copy = []
        for param_group in self.optimizer.param_groups:
            param_group_copy = []
            for parameters in param_group['params']:
                param_group_copy.append(torch.clone(parameters.data))
            self.param_groups_copy.append(param_group_copy)

    def _print_difference(self):
        delta_min = torch.inf
        delta_max = 0.0
        delta_sum = 0.0
        delta_n   = 1.0

        for i, param_group in enumerate(self.optimizer.param_groups):
            for j, parameters in enumerate(param_group['params']):
                delta = torch.sum(torch.abs(self.param_groups_copy[i][j] - parameters.data)).item()

                if delta_min > delta:
                    delta_min = delta
                if delta_max < delta:
                    delta_max = delta

                delta_sum += delta
                delta_n   += 1.0

                print(f'update ({i},{j}): {delta:15.10f}')

        print(f'update max :', delta_max)
        print(f'update min :', delta_min)
        print(f'update mean:', delta_sum / delta_n)
        print()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self, *args, **kwargs):
        return self.optimizer.state_dict(*args, **kwargs)

    def step(self, closure=None):
        self._copy_parameters()
        self.optimizer.step(closure=closure)
        self._print_difference()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

## ----------------------------------------------------------------------------

class LitTensorDataset(pl.LightningDataModule):
    def __init__(self, data : torch.utils.data.Dataset, val_size = 0.2, batch_size = 32, num_workers = 2):
        super().__init__()
        self.num_workers  = num_workers
        self.val_size     = val_size
        self.batch_size   = batch_size
        self.data         = data

    # This function is called by lightning trainer class with
    # the corresponding stage option
    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage == None:
            # Take a piece of the training data for validation
            self.data_train, self.data_val = torch.utils.data.random_split(self.data, [1.0 - self.val_size, self.val_size])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage == None:
            self.data_test = self.data

    # Custom method to create a data loader
    def get_dataloader(self, data):
        return torch.utils.data.DataLoader(data, batch_size=self.batch_size, num_workers=self.num_workers)

    # The following functions are called by the trainer class to
    # obtain data loaders
    def train_dataloader(self):
        return self.get_dataloader(self.data_train)

    def val_dataloader(self):
        return self.get_dataloader(self.data_val)

    def test_dataloader(self):
        return self.get_dataloader(self.data_test)

    def predict_dataloader(self):
        raise NotImplementedError

## ----------------------------------------------------------------------------

class LitModelWrapper(pl.LightningModule):
    def __init__(self,
                 # pytorch model class
                 model,
                 # Trainer options
                 patience_sd = 10, patience_es = 50, max_epochs = 1000, accelerator = 'gpu', devices = [0], strategy = 'auto', default_root_dir = 'checkpoints',
                 # Data options
                 val_size = 0.1, batch_size = 128, num_workers = 2,
                 # Learning rate
                 lr = 1e-3,
                 # Weight decay
                 weight_decay = 0.0,
                 # Other hyperparameters
                 betas = (0.9, 0.95), factor = 0.8,
                 # Optimizer and scheduler selection
                 scheduler = None, optimizer = 'Adam', optimizer_verbose = False, **kwargs):
        super().__init__()
        # Save all hyperparameters to `hparams` (e.g. lr)
        self.save_hyperparameters()
        self.train_loss        = []
        self.val_loss          = []
        self.optimizer         = optimizer
        self.optimizer_verbose = optimizer_verbose
        self.scheduler         = scheduler
        self.model             = model(**kwargs)

        self.trainer_options = {
            'patience_sd'      : patience_sd,
            'patience_es'      : patience_es,
            'max_epochs'       : max_epochs,
            'accelerator'      : accelerator,
            'devices'          : devices,
            'strategy'         : strategy,
            'default_root_dir' : default_root_dir,
        }
        self.data_options    = {
            'val_size'    : val_size,
            'batch_size'  : batch_size,
            'num_workers' : num_workers,
        }

    def configure_optimizers(self):
        # Get learning rates
        lr = self.hparams['lr']
        # Get weight_decay parameters
        weight_decay = self.hparams['weight_decay']

        # Initialize optimizer
        if   self.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=self.hparams['betas'])
        elif self.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay, betas=self.hparams['betas'])
        elif self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f'Unknown optimizer: {self.optimizer}')

        if self.optimizer_verbose:
            optimizer = LitVerboseOptimizer(optimizer)

        # Initialize scheduler
        if self.scheduler is None:
            scheduler = []
        elif self.scheduler == 'cycling':
            scheduler = [{'scheduler': torch.optim.lr_scheduler.CyclicLR(
                            optimizer,
                            base_lr        = 1e-4,
                            max_lr         = 5e-3,
                            step_size_up   = 10,
                            cycle_momentum = False,
                            verbose        = True),
                        'interval': 'epoch',
                        'monitor' : 'val_loss'
                        }]
        elif self.scheduler == 'plateau':
            scheduler = [{'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer,
                            patience = self.hparams['patience_sd'],
                            factor   = self.hparams['factor'],
                            mode     = 'min',
                            verbose  = True),
                        'interval': 'epoch',
                        'monitor' : 'train_loss'
                        }]
        else:
            raise ValueError(f'Unknown scheduler: {self.scheduler}')

        return [optimizer], scheduler

    def forward(self, x, **kwargs):
        return self.model.forward(x, **kwargs)

    def training_step(self, batch, batch_index):
        """Train model on a single batch"""
        X_batch = batch[0]
        y_batch = batch[1]
        # Call the model
        _, _, loss, loss_components = self.model.loss(X_batch, y_batch)
        # Send metrics to progress bar. We also don't want results
        # logged at every step, but let the logger accumulate the
        # results at the end of every epoch
        self.log(f'train_loss', loss.item(), prog_bar=True, on_step=False, on_epoch=True)
        for name, value in loss_components.items():
            self.log(f'train_{name}', value.item(), prog_bar=True, on_step=False, on_epoch=True)
        # Return whatever we might need in callbacks. Lightning automtically minimizes
        # the item called 'loss', which must be present in the returned dictionary
        return {'loss': loss}

    def validation_step(self, batch, batch_index):
        """Validate model on a single batch"""
        X_batch = batch[0]
        y_batch = batch[1]
        # Call the model
        _, _, loss, _ = self.model.loss(X_batch, y_batch)
        # Send metrics to progress bar. We also don't want results
        # logged at every step, but let the logger accumulate the
        # results at the end of every epoch
        self.log('val_loss', loss.item(), prog_bar=True, on_step=False, on_epoch=True)
        # Return whatever we might need in callbacks
        return {'val_loss': loss}

    def test_step(self, batch, batch_index):
        """Test model on a single batch"""
        X_batch = batch[0]
        y_batch = batch[1]
        x_hat, y_hat, loss, loss_components = self.model.loss(X_batch, y_batch)
        # Log whatever we want to aggregate later
        self.log('test_loss', loss)
        for name, value in loss_components.items():
            self.log(f'test_{name}', value.item(), on_step=False, on_epoch=True)
        # Return whatever we might need in callbacks
        return {'x': X_batch, 'x_hat': x_hat, 'y': y_batch, 'y_hat': y_hat, 'test_loss': loss}

    def predict_step(self, batch, batch_index):
        """Prediction on a single batch"""
        raise NotImplementedError

    def _setup_trainer_(self):
        self.trainer_matric_tracker      = LitMetricTracker()
        self.trainer_early_stopping      = pl.callbacks.EarlyStopping(monitor = 'train_loss', patience = self.trainer_options['patience_es'])
        self.trainer_checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k = 1, monitor = 'val_loss', mode = 'min')

        if os.path.exists(self.trainer_options['default_root_dir']):
            shutil.rmtree(self.trainer_options['default_root_dir'])

        # self.trainer is a pre-defined getter/setter in the LightningModule
        self.trainer = pl.Trainer(
            enable_checkpointing = True,
            logger               = False,
            enable_progress_bar  = True,
            max_epochs           = self.trainer_options['max_epochs'],
            accelerator          = self.trainer_options['accelerator'],
            devices              = self.trainer_options['devices'],
            strategy             = self.trainer_options['strategy'],
            default_root_dir     = self.trainer_options['default_root_dir'],
            callbacks            = [LitProgressBar(), self.trainer_early_stopping, self.trainer_checkpoint_callback, self.trainer_matric_tracker])

    def _train(self, data):

        if not isinstance(data, LitTensorDataset):
            data = LitTensorDataset(data, **self.data_options)

        # We always need a new trainer for training the model
        self._setup_trainer_()

        # Train model on train data. The fit method returns just None
        self.trainer.fit(self, data)

        # Get best model
        best_model = self.load_from_checkpoint(self.trainer_checkpoint_callback.best_model_path)
        # Lightning removes all training related objects before
        # saving the model. Recover all training components
        best_model.trainer                     = self.trainer
        best_model.trainer_matric_tracker      = self.trainer_matric_tracker
        best_model.trainer_early_stopping      = self.trainer_early_stopping
        best_model.trainer_checkpoint_callback = self.trainer_checkpoint_callback

        stats = {
            'best_val_error'  : self.trainer_checkpoint_callback.best_model_score.item(),
            'train_error'     : self.trainer_matric_tracker.train_error,
            'val_error'       : self.trainer_matric_tracker.val_error }

        return best_model, stats

    def _test(self, data):

        if not isinstance(data, LitTensorDataset):
            data = LitTensorDataset(data, **self.data_options)

        # We always need a new trainer for testing the model
        self._setup_trainer_()

        # Train model on train data. The test method returns accumulated
        # statistics sent to the logger
        stats = self.trainer.test(self, data)

        # There should only be one entry in stats
        assert len(stats) == 1

        # Get targets and predictions
        x, x_hat, y, y_hat = self.trainer_matric_tracker.test_predictions

        return x, x_hat, y, y_hat, stats[0]
