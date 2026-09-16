"""Official Kuzushiji-49 NPZ loader and EMNIST preprocessing."""
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url


class UprightEMNIST:
    def __call__(self, image):
        # The original IDX layout stores the character with x/y transposed.
        return image.transpose(Image.Transpose.TRANSPOSE)


class Kuzushiji49(Dataset):
    """CODH/NIJL Kuzushiji-49 (CC BY-SA 4.0), official train/test split."""
    base_url = 'http://codh.rois.ac.jp/kmnist/dataset/k49/'

    def __init__(self, root, train=True, download=False, transform=None):
        root = Path(root) / 'K49'
        partition = 'train' if train else 'test'
        arrays = []
        for kind in ('imgs', 'labels'):
            name = f'k49-{partition}-{kind}.npz'
            path = root / name
            if not path.exists():
                if not download:
                    raise FileNotFoundError(f'Missing {path}; enable download')
                root.mkdir(parents=True, exist_ok=True)
                temporary = name + '.part'
                # download_url treats an existing file without an MD5 as valid.
                # A previous interrupted partial download must not be reused.
                (root / temporary).unlink(missing_ok=True)
                download_url(self.base_url + name, str(root), filename=temporary)
                # Validate the archive before publishing a completed download.
                with np.load(root / temporary, allow_pickle=False) as archive:
                    archive['arr_0']
                (root / temporary).replace(path)
            with np.load(path, allow_pickle=False) as archive:
                arrays.append(archive['arr_0'])
        self.data, self.targets = arrays
        if self.data.ndim != 3 or self.data.shape[1:] != (28, 28) or len(self.data) != len(self.targets):
            raise ValueError('Invalid Kuzushiji-49 image/label dimensions')
        if self.targets.ndim != 1 or len(self.targets) == 0 or self.targets.min() < 0 or self.targets.max() >= 49:
            raise ValueError('Invalid Kuzushiji-49 labels')
        self.classes = [str(i) for i in range(49)]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        return self.transform(image) if self.transform else image, int(self.targets[index])
