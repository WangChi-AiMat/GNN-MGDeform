import time
import torch
import numpy as np
from MGDataset.MergedInMemoryDataset import MyDataset
from .BesselBasisLayer import BesselBasisLayer
from .gaussian_angle_expansion import gaussian_angle_expansion
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm


def load_dataset(dataset_path, Bessel_parameter=32, gaussian_parameter=8,
                 y_select=2, sample_index: int | None = None):


    print(f"loading {time.strftime('%H:%M:%S')}")
    Dataset_path = dataset_path

    dataset = MyDataset(root=Dataset_path)
    total_graphs = len(dataset)
    print(f" {total_graphs} graphs loaded")

    if sample_index is not None:
        if sample_index < 0 or sample_index >= total_graphs:
            raise ValueError(f"sample_index={sample_index}!  0~{total_graphs-1} ")
        print(f"{sample_index} ")
        dataset = [dataset[sample_index]]
    else:
        print("loading all graphs")

    sample = dataset[0]
    node_in_dim = sample.x.shape[1]
    edge_in_dim = sample.edge_attr.shape[1]

    num_nodes, y_dim = sample.y.shape
    if y_select is not None:
        if not isinstance(y_select, int):
            raise ValueError("y_select ")
        if y_select < 1 or y_select > y_dim:
            raise ValueError(f"y_select={y_select}! {y_dim}")

    bessel_layer = BesselBasisLayer(num_radial=Bessel_parameter, cutoff=4.5)
    new_data_list = []

    for data in tqdm(dataset, desc="edge expanding", ncols=60):

        # ---------- Step 1: Bessel----------
        dist = data.edge_attr[:, 0].unsqueeze(1)
        bessel_feat = bessel_layer(dist)

        # ---------- Step 2: Gaussian----------
        angles = data.edge_attr[:, 1:].cpu().numpy()
        valid_mask = angles > 0

        gaussian_list = []
        for row, mask in zip(angles, valid_mask):
            real_angles = row[mask]
            if len(real_angles) == 0:
                gaussian_list.append(np.zeros(gaussian_parameter, dtype=np.float32))
            else:
                gaussian_list.append(
                    gaussian_angle_expansion(real_angles, num_gauss=gaussian_parameter, gamma=10)
                )

        angle_feat = np.stack(gaussian_list, axis=0)

        # ---------- Step 3: ----------
        edge_attr_new = np.concatenate([bessel_feat, angle_feat], axis=1)
        data.edge_attr = torch.tensor(edge_attr_new, dtype=torch.float32)

        # ---------- Step 4: ----------
        if y_select is not None:
            selected_y = data.y[:, y_select - 1]
            data.y = selected_y.view(len(selected_y), 1)

        new_data_list.append(data)

    # ---InMemoryDataset ---
    class TempDataset(InMemoryDataset):
        def __init__(self, data_list):
            super().__init__(".")
            self.data, self.slices = self.collate(data_list)

    dataset = TempDataset(new_data_list)

    sample_after = dataset[0]
    edge_in_dim_after = sample_after.edge_attr.shape[1]

    return dataset, edge_in_dim_after
