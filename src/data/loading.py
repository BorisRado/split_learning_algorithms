from functools import partial
from hydra.utils import instantiate
from flwr_datasets import FederatedDataset


def _apply_transforms(batch, transforms):
  batch["img"] = [transforms(img) for img in batch["img"]]
  return batch


def get_dataset_from_cfg(dataset_cfg, partitioning_cfg, seed, client_idx):

    partitioner = instantiate(partitioning_cfg)
    fds = FederatedDataset(
        dataset=dataset_cfg.dataset_name,
        partitioners={"train": partitioner},
        seed=seed,
    )
    dataset = fds.load_partition(client_idx)
    transforms = instantiate(dataset_cfg.transforms)

    _apply_transforms_partial = partial(_apply_transforms, transforms=transforms)
    dataset = dataset.with_transform(_apply_transforms_partial)
    dataset = dataset.train_test_split(test_size=0.2, seed=seed)

    return dataset
