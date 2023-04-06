import json
import numpy as np

from pymatgen.core.structure import Structure 
from pymatgen.io.jarvis import JarvisAtomsAdaptor

from tqdm import tqdm


# ## Load up a data set
# 
# From the matbench dataset
data = '../../structures/Structures.json'
with open(data, "rb") as f:
    loaded = json.loads(f.read())
print(len(loaded))

# ### Set up to loop over the full dataset and build Alignn Inputs
# 
# This essentially replicates the `train_for_folder` function from [https://github.com/usnistgov/alignn/blob/main/alignn/train_folder.py](https://github.com/usnistgov/alignn/blob/main/alignn/train_folder.py)
dataset = []
for i in tqdm(range(len(loaded))):
    if loaded[i]['name'] == 'initial':
        info = {}
        info['jid'] = i
        info['target'] = np.random.randn() 
        structure = Structure.from_dict(loaded[i])
        structure = JarvisAtomsAdaptor.get_atoms(structure)
        info['atoms'] = structure.to_dict()
        dataset.append(info)

print(len(dataset))

# ### Now set up alignn model
from alignn.data import get_train_val_loaders
from jarvis.db.jsonutils import loadjson
from alignn.config import TrainingConfig

config = loadjson('config_git.json')
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


from alignn.train import train_dgl
train_dgl(
    config,
    train_val_test_loaders=[
        train_loader,
        val_loader,
        test_loader,
        prepare_batch,
    ],
)
