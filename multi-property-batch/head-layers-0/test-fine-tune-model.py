import json
import random
import torch

import numpy as np
import pandas as pd

from jarvis.core.graphs import Graph
from jarvis.core.atoms import Atoms

from pymatgen.io.jarvis import JarvisAtomsAdaptor
from pymatgen.core import Structure

#from alignn.models.alignn import ALIGNN
from alignn_multi import ALIGNN

#from alignn.data import get_torch_dataset
from data_multi import get_torch_dataset

from tqdm import tqdm

import dgl


def atoms_to_graph(atoms, cutoff=6.0, max_neighbors=12,
    atom_features="cgcnn", use_canonize=True):
    """Convert structure dict to DGLGraph."""
    #structure = Atoms.from_dict(atoms)
    structure = JarvisAtomsAdaptor.get_atoms(Structure.from_dict(atoms))
    return Graph.atom_dgl_multigraph(
        structure,
        cutoff=cutoff,
        atom_features=atom_features,
        max_neighbors=max_neighbors,
        compute_line_graph=True,
        use_canonize=use_canonize,
    )

def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]

def collate_line_graph(samples):
        """Dataloader helper to batch graphs cross `samples`."""
        graphs, line_graphs, has_prop, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_line_graph = dgl.batch(line_graphs)
        if len(labels[0].size()) > 0:
            return batched_graph, batched_line_graph, torch.stack(has_prop), torch.stack(labels)
        else:
            return batched_graph, batched_line_graph, torch.stack(has_prop), torch.tensor(labels)

### Parameters that we need to set
data = './dc_std_10_test.json'
data = '../../data/diels-test.json'
epochs = 0
load_prev_best_model = True
train_split = 90
test_split = 0
val_split = 10
batch_size = 4
lim = False      # Just used if you want a smaller subset for testing
learning_rate = 1e-3 
prop_indices = [5]  # This is the index of the property - taken from the OH column
n_early_stopping = 30
n_outputs=6 # Dimensions of the original MPR model, 6
n_hidden=0 # Number of hidden layers in the head
print_outputs=False
checkpoint_dir = './' # Where all your checkpoints will be saved
checkpoint_fp = 'checkpoint_ft_DC.pt' # The checkpoint of the general model to load up initially
### No need to edit beyond here



device = "cpu"
if torch.cuda.is_available():
    print('Found GPU and CUDA')
    device = torch.device("cuda")

with open(data, "rb") as f:
    dataset = json.loads(f.read())
if lim:
    dataset = dataset[:lim]

train_lim = int(len(dataset) / 100 * train_split)
val_lim = int(len(dataset) / 100 * val_split)
test_lim = int(len(dataset) / 100 * test_split)

print('Taining data: ', train_lim, 'Validaion data:  ', val_lim)

# Read in the old model
model = ALIGNN(n_outputs=n_outputs, 
        print_outputs=print_outputs, n_hidden=n_hidden)
model.to(device)
model.fc = torch.nn.Sequential(
    torch.nn.Linear(in_features=256, 
                    out_features=1,
                    bias=True)).to(device)
model.load_state_dict(torch.load(checkpoint_fp, map_location=torch.device(device))['model'])

# Set up the optimiser, loss and device

for datum in tqdm(dataset):
    datum['atoms'] = Atoms.to_dict(JarvisAtomsAdaptor.get_atoms(Structure.from_dict(datum['structure'])))
    datum['has_prop'] = torch.FloatTensor(datum['OH']) 
    datum['has_prop'] = datum['has_prop'][prop_indices]
    datum['target'] = torch.FloatTensor(datum['prop_list'])
    datum['target'] = datum['target'][prop_indices]

test_data = get_torch_dataset(dataset, target='target', neighbor_strategy="k-nearest", atom_features="cgcnn", line_graph=True)

for i in range(20):
    print(model((test_data.graphs[i], test_data.line_graphs[i], test_data.has_prop[i])).item(), test_data[i][-1])
