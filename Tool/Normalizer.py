import torch


class Normalizer():
    def __init__(self):
        self.mean = None
        self.std = None
        self._fitted = False

    def fit(self, data):

        if len(data.shape) != 2 or data.size(1) != 1:
            raise ValueError("[n_samples, 1]")

        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, unbiased=True, keepdim=True)

        self.std = torch.where(self.std > 0, self.std, torch.ones_like(self.std))

        self._fitted = True
        return self

    def transform(self, data):
        if not self._fitted:
            raise RuntimeError("fit")

        if len(data.shape) != 2 or data.size(1) != 1:
            raise ValueError("[n_samples, 1]")

        self.mean = self.mean.to(data.device)
        self.std = self.std.to(data.device)

        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if not self._fitted:
            raise RuntimeError("fit")

        if len(data.shape) != 2 or data.size(1) != 1:
            raise ValueError("[n_samples, 1]")

        self.mean = self.mean.to(data.device)
        self.std = self.std.to(data.device)

        return data * self.std + self.mean

    def get_scaling_params(self):
        if not self._fitted:
            raise RuntimeError("fit")

        return {
            'mean': self.mean.item(),
            'std': self.std.item()
        }

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std, '_fitted': self._fitted}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        self._fitted = state_dict['_fitted']