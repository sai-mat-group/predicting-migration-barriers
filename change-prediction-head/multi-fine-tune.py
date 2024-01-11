import json
import torch

from pymatgen.core.structure import Structure 
from pymatgen.io.jarvis import JarvisAtomsAdaptor

from tqdm import tqdm

from alignn.data import get_train_val_loaders
from jarvis.db.jsonutils import loadjson
from alignn.config import TrainingConfig
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn.train import train_dgl

data = '../trail/PM_90_10.json'  # The data to re-train on
model_path = './best_model.pt' # The model checkpoint to load initially
config_file = '../trail/config_git.json'
freeze_before = 2 # The layer at which to unfreeze the weights
output_features = 7

## Check for GPU and CUDA
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")

## Set up model configuration
model_config = ALIGNNConfig
model_config.output_features = output_features
model_config.classification = False
model_config.atom_input_features = 92
model_config.edge_input_features = 80
model_config.triplet_input_features = 40
model_config.embedding_features = 64
model_config.hidden_features = 256
model_config.name = "alignn"
model_config.alignn_layers = 4
model_config.gcn_layers = 4
model_config.link = "identity"
model_config.zero_inflated = False

model = ALIGNN(model_config)
model.load_state_dict(torch.load(model_path, map_location=torch.device(device))["model"])

## Redifine the fc module of the model
model.fc = torch.nn.Sequential(
    torch.nn.Linear(in_features=256,
                    out_features=1, # same number of output units as our number of classes
                    bias=True)).to(device)

## Load the trainig data
with open(data, "rb") as f:
    loaded = json.loads(f.read())

print((loaded[0]))
dataset = []
for i in tqdm(range(len(loaded))):
    info = {}
    info['jid'] = loaded[i]['jid']
    info['target'] = loaded[i]['target']
    info['atoms'] = loaded[i]['atoms']
    dataset.append(info)

## Set up the training configuration from the config file
config = loadjson(config_file)
config = TrainingConfig(**config)

(
    train_loader,
    val_loader,
    test_loader,
    prepare_batch,
) = get_train_val_loaders(
    dataset_array=dataset,
    target=config.target,
    n_train=config.n_train,
    n_val=config.n_val,
    n_test=config.n_test,
    train_ratio=config.train_ratio,
    val_ratio=config.val_ratio,
    test_ratio=config.test_ratio,
    batch_size=config.batch_size,
    atom_features=config.atom_features,
    neighbor_strategy=config.neighbor_strategy,
    standardize=config.atom_features != "cgcnn",
    id_tag=config.id_tag,
    pin_memory=config.pin_memory,
    workers=config.num_workers,
    save_dataloader=config.save_dataloader,
    use_canonize=config.use_canonize,
    filename=config.filename,
    cutoff=config.cutoff,
    max_neighbors=config.max_neighbors,
    output_features=config.model.output_features,
    classification_threshold=config.classification_threshold,
    target_multiplication_factor=config.target_multiplication_factor,
    standard_scalar_and_pca=config.standard_scalar_and_pca,
    keep_data_order=config.keep_data_order,
    output_dir=config.output_dir,
)


####### Freeze any layers in the model that we want to 
layer_count = 0
for child in model.children():
    print('#### CHILD ####')
    print(child)
    layer_count += 1
    if layer_count < freeze_before:
        for param in child.parameters():
            param.requires_grad = False

## Train
model.to(device)
train_dgl(
    config,
    model,
    train_val_test_loaders=[
        train_loader,
        val_loader,
        test_loader,
        prepare_batch,
    ],
)
