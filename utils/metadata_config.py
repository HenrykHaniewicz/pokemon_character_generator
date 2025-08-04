import torch
import torch.nn.functional as F
import random

class MetadataEncoder:
    def __init__(self, config):
        self.fields = config["metadata"]
        self.field_maps = {}
        self.field_dims = {}
        for key, values in self.fields.items():
            if not values:
                raise ValueError(f"Metadata field '{key}' has no options")
            if isinstance(values[0], bool):
                self.field_dims[key] = 1
            else:
                self.field_maps[key] = {v: i for i, v in enumerate(values)}
                self.field_dims[key] = len(values)

    @property
    def meta_dim(self):
        return sum(self.field_dims.values())

    def encode(self, meta):
        encoding = []
        for key, dim in self.field_dims.items():
            value = meta[key]
            if dim == 1:
                encoding.append(torch.tensor([1.0 if value else 0.0]))
            else:
                index = self.field_maps[key][value]
                one_hot = F.one_hot(torch.tensor(index), num_classes=dim).float()
                encoding.append(one_hot)
        return torch.cat(encoding)

def random_metadata(encoder):
    meta = {}
    for key, dim in encoder.field_dims.items():
        if dim == 1:
            meta[key] = random.choice([True, False])
        else:
            options = list(encoder.field_maps[key].keys())
            meta[key] = random.choice(options)
    return meta