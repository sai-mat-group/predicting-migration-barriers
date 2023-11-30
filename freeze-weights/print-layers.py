import json
import torch
import numpy as np
import os
import glob

from pymatgen.core.structure import Structure 
from pymatgen.io.jarvis import JarvisAtomsAdaptor

from tqdm import tqdm

data = 'fe_random_sample.json'  # The data to re-train on (gvrh)
model_path = 'checkpoint_500.pt' # The model checkpoint to load initially (std_phonons)
#freeze_before = 7 # The layer at which to unfreeze the weights

# ### Now set up alignn model
from alignn.data import get_train_val_loaders
from jarvis.db.jsonutils import loadjson
from alignn.config import TrainingConfig
from alignn.models.alignn import ALIGNN

from alignn.train import train_dgl
## Check for GPU and CUDA
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

model = ALIGNN()
layer_count = 0

for child in model.children():
    layer_count += 1
    if layer_count == 4:
        child[0].node_update.requires_grad = False ## Freeze 2 body
        child[0].edge_update.requires_grad = False ## Freeze 3 body
    if layer_count == 5:
        print(child[0])
        child[0].requires_grad = False # Freeze 2 body


