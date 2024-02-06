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

from torch.utils.data import DataLoader
#from alignn.data import get_torch_dataset
from data_multi import get_torch_dataset

from tqdm import tqdm

import dgl

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
data = '../../data/diels-test.json'
epochs = 1
batch_size = 1
lim = False      # Just used if you want a smaller subset for testing
prop_indices = [3]  # This is the index of the property - taken from the OH column
n_early_stopping = 30
n_outputs=6 # Dimensions of the original MPR model, 6
n_hidden=0 # Number of hidden layers in the head
print_outputs=False
checkpoint_dir = './fine-tune-diel/' # Where all your checkpoints will be saved
checkpoint = 'best_model.pt' # The checkpoint of the general model to load up initially
### No need to edit beyond here



device = "cpu"
if torch.cuda.is_available():
    print('Found GPU and CUDA')
    device = torch.device("cuda")

with open(data, "rb") as f:
    dataset = json.loads(f.read())
if lim:
    dataset = dataset[:lim]

for datum in tqdm(dataset):
    datum['atoms'] = Atoms.to_dict(JarvisAtomsAdaptor.get_atoms(Structure.from_dict(datum['structure'])))
    datum['has_prop'] = torch.FloatTensor(datum['OH']) 
    datum['has_prop'] = datum['has_prop'][prop_indices]
    datum['target'] = torch.FloatTensor(datum['prop_list'])
    datum['target'] = datum['target'][prop_indices]

test_data = get_torch_dataset(dataset, target='target', neighbor_strategy="k-nearest", atom_features="cgcnn", line_graph=True)

collate_fn = collate_line_graph
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Read in the old model
model_r= ALIGNN(n_outputs=n_outputs, 
        print_outputs=print_outputs, n_hidden=n_hidden)
model_r.to(device)
model_r.fc = torch.nn.Sequential(
    torch.nn.Linear(in_features=256, 
                    out_features=1,
                    bias=True)).to(device)
model_r.load_state_dict(torch.load(checkpoint_dir + checkpoint, map_location=torch.device(device))['model'])
#### DONT FORGET TO EVAL BEFORE RUNNING!!!
model_r.eval()

print('##### Testing after loading')
predictions = [('Predicted', 'True')]
with torch.no_grad():
    for dat in test_loader:
        g, lg, hp, target = dat
        out = model_r([g, lg, hp])
        predictions.append((out.item(), target.item()))

preds_test = pd.DataFrame(predictions)
preds_test.to_csv(checkpoint_dir + 'test-values.csv')
