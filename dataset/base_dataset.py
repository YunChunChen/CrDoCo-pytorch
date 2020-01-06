from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils import data


class BaseDataset(data.Dataset):

    def __init__(self, 
                 root, 
                 list_path, 
                 set_,
                 max_iters, 
                 image_size, 
                 labels_size, 
                 mean):

        self.root = Path(root)
        self.set = set_
        self.list_path = list_path.format(self.set)
        self.image_size = image_size
        self.mean = mean

        if labels_size is None:
            self.labels_size = self.image_size
        else:
            self.labels_size = labels_size

        with open(self.list_path) as f:
            self.img_ids = [i_id.strip() for i_id in f]

        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.files = []
        for name in self.img_ids:
            img_file, label_file = self.get_metadata(name)
            self.files.append((img_file, label_file, name))


    def __len__(self):
        return len(self.files)


    def preprocess(self, image):
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        return image.transpose((2, 0, 1))


    def get_image(self, image_path):
        return _load_img(image_path, self.image_size, Image.BICUBIC, rgb=True)


    def get_labels(self, label_path):
        return _load_img(label_path, self.labels_size, Image.NEAREST, rgb=False)


def _load_img(image_path, size, interpolation, rgb):
    img = Image.open(image_path)
    if rgb:
        img = img.convert('RGB')
    img = img.resize(size, interpolation)
    return np.asarray(img, np.float32)
