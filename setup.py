from setuptools import find_packages, setup

## ----------------------------------------------------------------------------

setup(
	name             = 'ixs',
	packages         = find_packages(),      
	version          = '0.0.1',
	description      = 'Forward and inverse models of small angle scattering',
	long_description = 'file: README.md',
	author           = 'Philipp Benner',
	author_email     = 'philipp.benner@bam.de',
	license          = 'MIT',
	classifiers      = [
		'Natural Language :: English',
		'Operating System :: OS Independent',
		'Programming Language :: Python :: 3',
	],
	install_requires = ['torch', 'pytorch_lightning', 'numpy', 'sasmodels', 'scipy', 'freia', 'matplotlib', 'seaborn', 'scikit-learn', 'pandas', 'h5py', 'dill'],
	python_requires  = '>3.8'
)
