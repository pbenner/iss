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

import numpy as np
import torch
import pytorch_lightning as pl
import sys

from copy   import deepcopy
from tqdm   import tqdm
from typing import Optional

from sklearn.model_selection import KFold

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

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    self.train_error_batch.append(outputs['loss'].item())

  def on_train_epoch_end(self, *args, **kwargs):
    self.train_error.append(torch.mean(torch.tensor(self.train_error_batch)))
    self.train_error_batch = []

  def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    self.val_error_batch.append(outputs['val_loss'].item())

  def on_validation_epoch_end(self, trainer, pl_module):
    self.val_error.append(torch.mean(torch.tensor(self.val_error_batch)))
    self.val_error_batch = []

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
    def __init__(self, data : torch.utils.data.TensorDataset, n_splits = 1, val_size = 0.2, batch_size = 32, num_workers = 2, shuffle = True, random_state = 42):
        super().__init__()
        self.num_workers  = num_workers
        self.val_size     = val_size
        self.batch_size   = batch_size
        self.data         = data
        # Setup k-fold cross-validation if n_splits > 1
        self.n_splits     = n_splits
        if n_splits > 1:
            self.splits   = list(KFold(n_splits, shuffle = shuffle, random_state = random_state).split(self.data.X, self.data.y))

    # Custom method to set the fold for cross-validation
    def setup_fold(self, k):
        if self.n_splits < 2:
            raise ValueError(f'k-fold cross-validation is not available, because n_splits is set to {self.n_splits}')
        self.k = k

    # This function is called by lightning trainer class with
    # the corresponding stage option
    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage == None:
            # Check if we are using cross-validation
            if self.n_splits > 1:
                train_index, _ = self.splits[self.k]
                self.data_train = torch.utils.data.Subset(self.data, train_index)
            else:
                self.data_train = self.data
            # Take a piece of the training data for validation
            self.data_train, self.data_val = torch.utils.data.random_split(self.data_train, [1.0 - self.val_size, self.val_size])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage == None:
            # Check if we are using cross-validation
            if self.n_splits > 1:
                _, test_index = self.splits[self.k]
                self.data_test = torch.utils.data.Subset(self.data, test_index)
            else:
                self.data_test = self.data

        # Assign predict dataset for use in dataloader(s)
        if stage == 'predict' or stage == None:
            self.data_predict = self.data

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
        return self.get_dataloader(self.data_predict)

## ----------------------------------------------------------------------------

class LitModelWrapper(pl.LightningModule):
    def __init__(self,
                 # pytorch model class
                 model,
                 # Trainer options
                 patience_sd = 10, patience_es = 50, max_epochs = 1000, accelerator = 'gpu', devices = [0], strategy = 'auto',
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

        self.lit_trainer         = None
        self.lit_trainer_options = {
            'patience_sd' : patience_sd,
            'patience_es' : patience_es,
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

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_index):
        """Train model on a single batch"""
        X_batch = batch[0]
        y_batch = batch[1]
        loss, loss_components = self.model.__train_step__(X_batch, y_batch)
        self.log(f'train_loss', np.log10(loss.item()), on_step=False, on_epoch=True, prog_bar=True, logger=False)
        for name, value in loss_components.items():
            self.log(f'train_{name}', np.log10(value.item()), on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return {'loss': loss}

    def validation_step(self, batch, batch_index):
        """Validate model on a single batch"""
        X_batch = batch[0]
        y_batch = batch[1]
        _, loss, _ = self.model.__test_step__(X_batch, y_batch)
        self.log('val_loss', np.log10(loss.item()), on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return {'val_loss': loss}

    def test_step(self, batch, batch_index):
        """Test model on a single batch"""
        X_batch = batch[0]
        y_batch = batch[1]
        y_hat, _, _ = self.model.__test_step__(X_batch, y_batch)
        # Return predictions
        return {'y': y_batch[:,0].detach().cpu(), 'y_hat': y_hat[:,0].detach().cpu()}

    def test_epoch_end(self, test_step_outputs):
        # Collect predictions from individual batches
        y     = torch.tensor([])
        y_hat = torch.tensor([])
        for output in test_step_outputs:
            y     = torch.cat((y    , output['y']))
            y_hat = torch.cat((y_hat, output['y_hat']))
        # Save predictions for evaluation
        self.test_y     = y
        self.test_y_hat = y_hat

    def predict_step(self, batch, batch_index):
        """Prediction on a single batch"""
        return self.forward(batch[0])

    def _setup_trainer_(self):
        self.lit_matric_tracker      = LitMetricTracker()
        self.lit_early_stopping      = pl.callbacks.EarlyStopping(monitor = 'val_loss', patience = self.lit_trainer_options['patience_es'])
        self.lit_checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k = 1, monitor = 'val_loss', mode = 'min')

        self.lit_trainer = pl.Trainer(
            enable_checkpointing = True,
            logger               = False,
            enable_progress_bar  = True,
            max_epochs           = self.lit_trainer_options['max_epochs'],
            accelerator          = self.lit_trainer_options['accelerator'],
            devices              = self.lit_trainer_options['devices'],
            strategy             = self.lit_trainer_options['strategy'],
            callbacks            = [LitProgressBar(), self.lit_early_stopping, self.lit_checkpoint_callback, self.lit_matric_tracker])

    def _train(self, data):

        data = LitTensorDataset(data, **self.lit_data_options)

        # We always need a new trainer for training the model
        self._setup_trainer_()

        # Train model on train data and use validation data for early stopping
        self.lit_trainer.fit(self, data)

        # Get best model
        self.lit_model = self.load_from_checkpoint(self.lit_checkpoint_callback.best_model_path)

        result = {
            'best_val_error': self.lit_checkpoint_callback.best_model_score.item(),
            'train_error'   : torch.stack(self.lit_matric_tracker.train_error).tolist(),
            'val_error'     : torch.stack(self.lit_matric_tracker.val_error  ).tolist() }

        return result

    def _predict(self, data):

        data = LitTensorDataset(data, **self.lit_data_options)

        if self.lit_trainer is None:
            self._setup_trainer_()

        y_hat_batched = self.lit_trainer.predict(self, data)

        return torch.cat(y_hat_batched, dim=0)

    def _cross_validation(self, data, n_splits, shuffle = True, random_state = 42):

        data = LitTensorDataset(data, n_splits = n_splits, shuffle = shuffle, random_state = random_state, **self.lit_data_options)

        if data.n_splits < 2:
            raise ValueError(f'k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits={n_splits}')

        y_hat = torch.tensor([], dtype = torch.float)
        y     = torch.tensor([], dtype = torch.float)

        initial_model = self.model

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
