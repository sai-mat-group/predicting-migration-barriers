# predicting-migration-barriers

Repository for migration barrier dataset which will be used for model constructions.

## Instructions to set up ALIGNN


We build a conda environment (here I use mamba, as it is much more efficient, but if you have conda you can just substitute `conda` for `mamba` below) 
```
mamba create --name alignn
mamba activate alignn
```

Now we want to install `alignn`, I usually have an `src` directory in my home for any install from source packages

```
cd 
cd src
git clone https://github.com/usnistgov/alignn.git
cd alignn
python setup.py develop
pip install dgl-cu111
```

We are almost installed. One issue is that `alignn` is pinned to `pytorch-1.12` - but this is not compatible with the A100 GPUs. We find that upgrading `pytorch` results in a few warnings and errors during install, but still seems to run as expected. You don't need to do this unless you are using nice GPUs :)

```
pip install --upgrade torch
```

If you want to use any `pymatgen` functionality

```
pip install pymatgen
```
