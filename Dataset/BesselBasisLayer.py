import torch
import torch.nn as nn
import math


class BesselBasisLayer(nn.Module):
    def __init__(self, num_radial, cutoff, include_dist=False):
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.include_dist = include_dist

        self.freq = nn.Parameter(
            torch.arange(1, num_radial + 1, dtype=torch.float32) * math.pi / cutoff,
            requires_grad=False
        )

    def cutoff_fn(self, dist):

        return 0.5 * (torch.cos(math.pi * dist / self.cutoff) + 1.0) * (dist < self.cutoff)

    def forward(self, dist):

        dist = dist.view(-1, 1)

        # Bessel
        bessel = torch.sin(self.freq * dist) / dist  # [num_edges, num_radial]
        bessel = bessel * math.sqrt(2.0 / self.cutoff)

        # cutoff
        bessel = bessel * self.cutoff_fn(dist)

        if self.include_dist:
            out = torch.cat([bessel, dist], dim=-1)  # [num_edges, num_radial+1]
        else:
            out = bessel  # [num_edges, num_radial]

        return out
