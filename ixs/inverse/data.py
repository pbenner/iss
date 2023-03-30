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
import h5py
import os

from sklearn.preprocessing import StandardScaler

## ----------------------------------------------------------------------------

class ScatteringData(torch.utils.data.TensorDataset):

    def __init__(self, path, shapes, input_keys, ndim_pad_x, ndim_y, ndim_z, ndim_pad_zy, target = 'I'):
    
        self.ndim_pad_x  = ndim_pad_x
        self.ndim_y      = ndim_y
        self.ndim_z      = ndim_z
        self.ndim_pad_zy = ndim_pad_zy 

        inputs, labels, shapes_dict = self.read_data(path, shapes, input_keys, target)

        self.shapes_dict = shapes_dict

        super().__init__(inputs, labels)

    def read_data(self, path, shapes, input_keys, target):
        """
        Load the trainig data from HDF files:
        Arguments:
            path(str): a directory with all HDF available for training
            shapes(int): total number of shapes present in the training set
            input_keys(list): a list naming all the parameters a network is supposed to identify, named exatly as in the HDF trainung files and starting with 'shape' and 'radius'
        """

        if not(input_keys[0] == 'shape' and input_keys[1] == 'radius'):
            raise ValueError('The first two parameters of input keys should be named "shape" and "radius"')

        self.input_features = input_keys.copy()
        files = [ f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) ]
        self.ndim_x = shapes*2 + len(input_keys)-2

        # Check dimensions
        assert self.ndim_x + self.ndim_pad_x == self.ndim_y + self.ndim_z + self.ndim_pad_zy, "Dimensions don't match up"

        labels = torch.zeros(len(files), self.ndim_y, dtype = torch.float32)
        inputs = torch.zeros(len(files), self.ndim_x, dtype = torch.float32)
        shapes_dict = {}

        for i, f in enumerate(files):
            with h5py.File(os.path.join(path, f), 'r') as file:

                print(f'Reading: {file}', end = '\r')

                assert self.ndim_y ==  torch.from_numpy(file['entry/I'][()].flatten()).shape[0], "scattering curve has different size"

                # Read I or I_noisy here, specified by target variable
                labels[i,:] = torch.from_numpy(file[f'entry/{target}'][()].flatten())
                for i_k, key in enumerate(input_keys):
                    try:
                        if key == 'shape':
                            shape = file['properties'][key][()].decode("utf-8")
                            value = torch.zeros(shapes)
                            if shape not in shapes_dict:
                                shapes_dict[shape] = max(list(shapes_dict.values()))+1 if len(shapes_dict.keys()) > 0 else 0
                            value[shapes_dict[shape]] = 1
                            inputs[i, i_k:i_k+shapes] = value
                        elif key == 'radius':
                            value = torch.zeros(shapes)
                            value[shapes_dict[shape]] = file['properties'][key][()]
                            inputs[i, i_k+(shapes-1):i_k+(shapes*2-1)] = value
                        else:
                            inputs[i, i_k+(shapes*2-2)] = file['properties'][key][()]
                    except KeyError:
                        # e.g spheres don't have all of the properties a cylinder does
                        pass

        return inputs, labels, shapes_dict

    def normalize(self, scaler):

        n_shapes = len(self.shapes_dict.keys())

        # Get input data
        x = self.tensors[0]
        # Split data matrix into left and right part
        x_left  = x[:,:n_shapes ]
        x_right = x[:, n_shapes:]

        # Fit and apply scaler
        scaler.fit(x[:, n_shapes:])

        x_tmp = torch.from_numpy(scaler.transform(x_right))
        x_tmp = torch.concatenate((x_left, x_tmp), axis=1).type(torch.float32)

        self.tensors[0].set_(x_tmp)
