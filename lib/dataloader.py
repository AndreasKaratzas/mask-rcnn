
import os
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.images = list(
            sorted(os.listdir(os.path.join(root_dir, "images"))))
        self.masks = list(
            sorted(os.listdir(os.path.join(root_dir, "masks"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "images", self.images[idx])
        mask_path = os.path.join(self.root_dir, "masks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert('L')

        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            if xmin == xmax:
                xmin -= 1
                xmax += 1

            if ymin == ymax:
                ymin -= 1
                ymax += 1

            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = []
        for label in obj_ids:
            if label == 6:
                labels.append(1)
            elif label == 7:
                labels.append(2)
            elif label == 10:
                labels.append(3)
            else:
                raise ValueError("Unknown label found")

        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        if boxes.ndim == 1:
            boxes = boxes.unsqueeze(0)

        if labels.ndim < 1:
            labels = labels.unsqueeze(0)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id, "area": area,
                  "iscrowd": iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)


class TestDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.images = list(
            sorted(os.listdir(os.path.join(root_dir, "images"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, "images", self.images[idx])

        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img=img)

        return img

    def __len__(self):
        return len(self.images)
