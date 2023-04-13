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

import torch

## ----------------------------------------------------------------------------

class TorchStandardScaler(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.tensor(dim*[0.0]), requires_grad=False)
        self.std  = torch.nn.Parameter(torch.tensor(dim*[1.0]), requires_grad=False)

    def fit(self, x):
        self.mean = torch.nn.Parameter(x.mean(0, keepdim = False                  )       , requires_grad=False)
        self.std  = torch.nn.Parameter(x.std (0, keepdim = False, unbiased = False) + 1e-8, requires_grad=False)

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x*self.std + self.mean

## ----------------------------------------------------------------------------

class SASScaler(torch.nn.Module):

    def __init__(self, dim_input, dim_output, shapes_dict):

        super().__init__()
        self.dim_input     = dim_input
        self.dim_output    = dim_output
        self.n_shapes      = len(shapes_dict.keys())

        self.scaler_input  = TorchStandardScaler(dim_input - self.n_shapes)
        self.scaler_output = None

    def fit_input(self, x):

        # Get only the part of the data that we want to scale
        x_right = x[:, self.n_shapes:self.dim_input]

        # Fit and apply scaler
        self.scaler_input.fit(x_right)

    def fit_output(self, y):

        pass

    def normalize_input(self, x):

        # Split data matrix into left, right and padded part
        x_left   = x[:,:self.n_shapes ]
        x_right  = x[:, self.n_shapes:self.dim_input]
        x_pad    = x[:, self.dim_input:]
        # Scale only the right part
        x_scaled = self.scaler_input.transform(x_right)

        return torch.concatenate((x_left, x_scaled, x_pad), axis=1)

    def denormalize_input(self, x):

        # Split data matrix into left, right and padded part
        x_left   = x[:,:self.n_shapes ]
        x_right  = x[:, self.n_shapes:self.dim_input]
        x_pad    = x[:, self.dim_input:]
        # Scale only the right part
        x_scaled = self.scaler_input.inverse_transform(x_right)

        return torch.concatenate((x_left, x_scaled, x_pad), axis=1)

    def normalize_output(self, y):

        return y

    def denormalize_output(self, y):

        return y
