import json
import torch
import csv

from pymatgen.core.structure import Structure 
from pymatgen.io.jarvis import JarvisAtomsAdaptor

from jarvis.core.graphs import Graph
from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import loadjson
from alignn.config import TrainingConfig
from alignn.models.alignn import ALIGNN
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error

def atoms_to_graph(atoms, cutoff, max_neighbors,
    atom_features, use_canonize):
    """Convert structure dict to DGLGraph."""
    structure = Atoms.from_dict(atoms)
    return Graph.atom_dgl_multigraph(
        structure,
        cutoff=cutoff,
        atom_features=atom_features,
        max_neighbors=max_neighbors,
        compute_line_graph=True,
        use_canonize=use_canonize,
    )

### These 3 lines should be edited
config = loadjson('config.json')
data = 'test-set.json'  # The data to test on
model_path = '../trail/checkpoint-eform.pt' # The model checkpoint to load initially
### No more editing required

with open(data, "rb") as f:
    loaded = json.loads(f.read())

dataset = []
for i in tqdm(range(len(loaded))):
    info = {}
    info['jid'] = loaded[i]['jid']
    info['target'] = loaded[i]['target']
    info['atoms'] = loaded[i]['atoms']
    dataset.append(info)

config = TrainingConfig(**config)



device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


model = ALIGNN()
model.fc = torch.nn.Sequential(
    torch.nn.Linear(in_features=256, 
                    out_features=64, # same number of output units as our number of classes
                    bias=True),
    torch.nn.Linear(in_features=64,
                    out_features=1, # same number of output units as our number of classes
                    bias=True)).to(device)

model.load_state_dict(torch.load(model_path, map_location=torch.device(device))["model"])

results = []
ground_truth = []
ignored = 0

for info in dataset:
    try:
        g, lg = atoms_to_graph(info['atoms'],
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors,
            atom_features=config.atom_features,
            use_canonize=config.use_canonize
            )

        results.append(model([g.to(device), lg.to(device)])
            .detach().cpu().numpy()
            .flatten()[0])
        ground_truth.append(info['target'])
    except:
        print('Case did not run: ', info['jid'])
        ignored = ignored + 1

list_zip = zip(ground_truth, results)

print('r2: {0:4f}'.format(r2_score(ground_truth, results)))
print('mae: {0:4f}'.format(mean_absolute_error(ground_truth, results)))
print('Ignored {} cases'.format(ignored))

with open('predictions.csv', 'w') as f:
    # Create a CSV writer object that will write to the file 'f'
    csv_writer = csv.writer(f)
    # Write all of the rows of data to the CSV file
    csv_writer.writerows(list_zip)
