import json

from pymatgen.core.structure import Structure 
from pymatgen.io.jarvis import JarvisAtomsAdaptor

from tqdm import tqdm

data = 'Structures.json'
MAX_ATOMS = 300 # Maximum number of atoms allowed in a structure
with open(data, "rb") as f:
    loaded = json.loads(f.read())

dataset = []
for i in tqdm(range(len(loaded))):
    if (i+3)%3 == 0: # Only take the first structure i.e. initial state, not transition or final
        info = {}
        info['jid'] = int(i)
        info['target'] = loaded[i]['barrier']
        structure = Structure.from_dict(loaded[i])
        if len(structure) <= MAX_ATOMS: # Check if length is the problem for some graphs
            structure = JarvisAtomsAdaptor.get_atoms(structure)
            info['atoms'] = structure.to_dict()
            dataset.append(info)


for i in list_to_drop:
    dataset.pop(int(i))

# ### Now set up alignn model
from alignn.data import get_train_val_loaders
from jarvis.db.jsonutils import loadjson
from alignn.config import TrainingConfig

config = loadjson('../bin/config_git.json')
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
