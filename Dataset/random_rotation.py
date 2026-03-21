import torch

def random_rotation(edge_attr):
    perm = torch.randperm(3)
    edge_attr_rot = edge_attr[:, perm]

    symmetry = torch.randint(0, 2, (3,)) * 2 - 1
    symmetry = symmetry.to(edge_attr.device)
    edge_attr_rot = edge_attr_rot * symmetry

    return edge_attr_rot