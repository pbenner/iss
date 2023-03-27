from setuptools import find_packages, setup

## ----------------------------------------------------------------------------

setup(
	name             = 'ixs',
	packages         = find_packages(),      
    version          = '0.0.1',
	description      = 'Forward and inverse models of small angle scattering',
	long_description = 'file: README.md',
	author           = 'Sofya Laskina',
	author_email     = 'sofyalaskina@gmail.com',
	license          = 'MIT',
	classifiers      = [
		'Natural Language :: English',
		'Operating System :: OS Independent',
		'Programming Language :: Python :: 3',
	],
	install_requires = ['torch', 'numpy', 'sasmodels', 'scipy', 'freia', 'mp-pyrho', 'scikit-learn'],
	python_requires  = '>3.8'
)
