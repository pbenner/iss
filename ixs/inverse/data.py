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
import h5py
import os

from collections import UserDict
from sklearn.model_selection import train_test_split

## ----------------------------------------------------------------------------

class ScatteringMetaData(UserDict):

    def __init__(self, qx, shapes_dict, input_features, ndim_x, ndim_pad_x, ndim_y, ndim_z, ndim_pad_zy):

        super().__init__()
        # Set config values, use method from super class to add
        # new items
        super().__setitem__('qx',             qx)
        super().__setitem__('shapes_dict',    shapes_dict)
        super().__setitem__('input_features', input_features)
        super().__setitem__('ndim_x',         ndim_x)
        super().__setitem__('ndim_pad_x',     ndim_pad_x)
        super().__setitem__('ndim_y',         ndim_y)
        super().__setitem__('ndim_z',         ndim_z)
        super().__setitem__('ndim_pad_zy',    ndim_pad_zy)

    def __setitem__(self, key, value):

        if key not in self:
            raise KeyError(f'"{key}" is an invalid parameter')

        super().__setitem__(key, value)

## ----------------------------------------------------------------------------

class ScatteringData(torch.utils.data.TensorDataset):

    def __init__(self, inputs, outputs, qx, shapes_dict, input_features, ndim_x, ndim_pad_x, ndim_y, ndim_z, ndim_pad_zy):
    
        self.ndim_x         = ndim_x
        self.ndim_pad_x     = ndim_pad_x
        self.ndim_y         = ndim_y
        self.ndim_z         = ndim_z
        self.ndim_pad_zy    = ndim_pad_zy
        self.shapes_dict    = shapes_dict   .copy()
        self.input_features = input_features.copy()
        self.qx             = qx

        super().__init__(inputs, outputs)

    def __new_data__(self, inputs, outputs):
        return ScatteringData(inputs, outputs, self.qx, self.shapes_dict, self.input_features, self.ndim_x, self.ndim_pad_x, self.ndim_y, self.ndim_z, self.ndim_pad_zy)

    @classmethod
    def load_from_dir(cls, path, n_shapes, input_features, ndim_pad_x, ndim_y, ndim_z, ndim_pad_zy, target = 'I', logI = True):
        """
        Load the trainig data from HDF files. The result is a data set of inputs (parameters) and outputs (scattering curves).
        The matrix of inputs encodes shapes using one-hot-encoding, i.e. the first n_shapes columns determine the type of shape,
        where n_shapes is the number of shapes in the data set. In addition, each shape has a separate column for the radius.
        All other parameters are sharing the same column for all shapes. Hence, the first 2*n_shapes columns are reserved for
        shape and radius information, and the rest corresponds to the remaining features.

        Arguments:
            path(str): a directory with all HDF available for training
            shapes(int): total number of shapes present in the training set
            input_features(list): a list naming all the parameters a network is supposed to identify, named exatly as in the HDF trainung files and starting with 'shape' and 'radius'
        """

        if not(input_features[0] == 'shape' and input_features[1] == 'radius'):
            raise ValueError('The first two parameters of input keys should be named "shape" and "radius"')

        files = [ f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) ]
        ndim_x = n_shapes*2 + len(input_features)-2
        qx = None

        # Check dimensions
        assert ndim_x + ndim_pad_x == ndim_y + ndim_z + ndim_pad_zy, "Dimensions don't match up"

        outputs = torch.zeros(len(files), ndim_y, dtype = torch.float32)
        inputs  = torch.zeros(len(files), ndim_x, dtype = torch.float32)
        shapes_dict = {}

        for i, f in enumerate(files):
            with h5py.File(os.path.join(path, f), 'r') as file:

                print(f'Reading: {file}', end = '\r')

                assert ndim_y == torch.from_numpy(file[f'entry/{target}'][()].flatten()).shape[0], "scattering curve has different size"

                if qx is None:
                    qx = torch.from_numpy(file[f'entry/qx'][()])
                else:
                    assert torch.mean((qx - torch.from_numpy(file[f'entry/qx'][()]))**2).item() < 1e-8, "qx values differ"

                # Read I or I_noisy here, specified by target variable
                outputs[i,:] = torch.from_numpy(file[f'entry/{target}'][()].flatten())
                if logI:
                    outputs[i,:] = torch.log(outputs[i,:])
                # Read input parameters
                for i_k, key in enumerate(input_features):
                    try:
                        if key == 'shape':
                            shape = file['properties'][key][()].decode("utf-8")
                            value = torch.zeros(n_shapes)
                            if shape not in shapes_dict:
                                shapes_dict[shape] = max(list(shapes_dict.values()))+1 if len(shapes_dict.keys()) > 0 else 0
                            value[shapes_dict[shape]] = 1
                            inputs[i, i_k:i_k+n_shapes] = value
                        elif key == 'radius':
                            value = torch.zeros(n_shapes)
                            value[shapes_dict[shape]] = file['properties'][key][()]
                            inputs[i, i_k+(n_shapes-1):i_k+(n_shapes*2-1)] = value
                        else:
                            inputs[i, i_k+(n_shapes*2-2)] = file['properties'][key][()]
                    except KeyError:
                        # e.g spheres don't have all of the properties a cylinder does
                        pass

        return ScatteringData(inputs, outputs, qx, shapes_dict, input_features, ndim_x, ndim_pad_x, ndim_y, ndim_z, ndim_pad_zy)

    def subset(self, index):

        inputs, outputs = self[index]

        return self.__new_data__(inputs, outputs)

    def train_test_split(self, test_size = 0.1, shuffle = True, random_state = 42):

        idx_train, idx_test = train_test_split(range(len(self)), test_size = test_size, shuffle = shuffle, random_state = random_state)

        return self.subset(idx_train), self.subset(idx_test)

    @property
    def metadata(self):
        return ScatteringMetaData(self.qx, self.shapes_dict, self.input_features, self.ndim_x, self.ndim_pad_x, self.ndim_y, self.ndim_z, self.ndim_pad_zy)

    @property
    def X(self):
        return self.tensors[0]

    @property
    def y(self):
        return self.tensors[1]

    @classmethod
    def load(cls, filename : str) -> 'ScatteringData':

        with open(filename, 'rb') as f:
            data = dill.load(f)

        if not isinstance(data, cls):
            raise ValueError(f'file {filename} contains incorrect model class {type(data)}')

        return data

    def save(self, filename : str) -> None:

        with open(filename, 'wb') as f:
            dill.dump(self, f)
