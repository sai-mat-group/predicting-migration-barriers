# predicting-migration-barriers

Repository for migration barrier dataset which will be used for model constructions.

## Instructions to set up ALIGNN


We build a conda environment (here I use mamba, as it is much more efficient, but if you have conda you can just substitute `conda` for `mamba` below) 
```
mamba create --name alignn python=3.8
mamba activate alignn
```

Now we want to install `alignn`, I usually have an `src` directory in my home for any install from source packages
If you already cloned `alignn`, skip this block

```
cd 
cd src
git clone https://github.com/usnistgov/alignn.git
```

Go to the `alignn` directory and make sure all is up to date
```
cd alignn
git pull
python setup.py develop
pip install dgl-cu111
```

If you want to use any `pymatgen` functionality:

```
pip install pymatgen
```
