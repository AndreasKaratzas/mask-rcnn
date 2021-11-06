
import os
import torch
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as F

from typing import List
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from torchvision.utils import make_grid, draw_bounding_boxes, draw_segmentation_masks


class Visual:
    def __init__(self, model: torch.nn.Module, root_dir: str, device: torch.device, conf_threshold: float,
                 prob_threshold: float = 0.5, batch_size: int = 1, raw_root_dir: bool = False):
        super().__init__()

        matplotlib.style.use('ggplot')
        plt.rcParams["savefig.bbox"] = 'tight'

        self.model = model
        self.root_dir = root_dir
        self.device = device
        self.conf_threshold = conf_threshold
        self.prob_threshold = prob_threshold
        self.batch_size = batch_size
        self.raw_root_dir = raw_root_dir
        # enter evaluation mode
        self.model.eval()

    def show(self, img_set):
        if not isinstance(img_set, list):
            img_set = [img_set]

        fig, axs = plt.subplots(ncols=len(img_set), squeeze=False)

        for i, img in enumerate(img_set):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.show()

    def crop_transform(self, crop_size: int):
        return T.CenterCrop(crop_size)

    def img_grid(self, img_list: List[np.ndarray]):
        grid = make_grid(img_list)
        return grid

    def grouped(self, iterable):
        return zip(*[iter(iterable)] * self.batch_size)

    def build_batch(self):
        if not self.raw_root_dir:
            self.img_dir = os.path.join(self.root_dir, "images", "test")
        else:
            self.img_dir = self.root_dir

        self.img_list = list(sorted(os.listdir(self.img_dir)))
        self.imgs = []

        for img_subset in self.grouped(self.img_list):
            batch = []
            global_min = 4096

            for img_instance in img_subset:
                img_instance = read_image(
                    os.path.join(self.img_dir, img_instance))
                instance_min = torch.min(torch.tensor(img_instance.shape[1:])).item()

                if global_min > instance_min:
                    global_min = instance_min

            crop_size = global_min
            transform = self.crop_transform(crop_size)

            for img_instance in img_subset:
                img_instance = read_image(
                    os.path.join(self.img_dir, img_instance))
                batch.append(transform(img_instance))

            self.imgs.append(batch)

    @torch.no_grad()
    def test_model(self, alpha: float = 0.7, color:  str = 'green'):
        for batch in self.imgs:
            # This expects a numpy.ndarray of value range in [0, 255]
            raw_batch = torch.stack(batch)
            # This preprocesses raw batch
            batch = convert_image_dtype(raw_batch, dtype=torch.float).to(self.device)
            # Feed input to model
            outputs = self.model(batch)
            # Filter predictions with respecto to given thresholds
            boolean_masks = [
                out['masks'][out['scores'] > self.conf_threshold] > self.prob_threshold
                for out in outputs
            ]
            # Configure colors
            colors = [color] * len(boolean_masks)
            # Draw predicted masks on image
            vis_result = [
                draw_segmentation_masks(img, mask.squeeze(1), colors=colors, alpha=alpha)
                for img, mask in zip(raw_batch, boolean_masks)
            ]
            # Display the result
            self.show(vis_result)


class VisualTest:
    def __init__(self):
        super().__init__()

        matplotlib.style.use('ggplot')
        plt.rcParams["savefig.bbox"] = 'tight'

    def draw_segm_masks(self, img: np.ndarray, masks: torch.Tensor, color: str = "green", alpha: float = 0.7):
        return draw_segmentation_masks(img, masks, colors=[color] * masks.shape[0], alpha=alpha)

    def show(self, img_set):
        if not isinstance(img_set, list):
            img_set = [img_set]

        fig, axs = plt.subplots(ncols=len(img_set), squeeze=False)

        for i, img in enumerate(img_set):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.show()

    def visualize(self, img: torch.Tensor, masks: torch.Tensor):
        img = img.clone().detach().type(torch.ByteTensor)
        masks = masks.clone().detach().type(torch.BoolTensor)
        img_set = self.draw_segm_masks(img=img, masks=masks)
        self.show(img_set=img_set)
