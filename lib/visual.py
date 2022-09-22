
import os
import torch
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as F

from typing import List
from pathlib import Path
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
from torchvision.utils import make_grid, draw_bounding_boxes, draw_segmentation_masks


class Visual:
    def __init__(self, model: torch.nn.Module, root_dir: str, device: torch.device, conf_threshold: float,
                 prob_threshold: float = 0.5, batch_size: int = 1, raw_root_dir: bool = False, num_classes: int = 1):
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
        self.num_classes = num_classes

        self.colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                       for i in range(self.num_classes)]

        # enter evaluation mode
        self.model.eval()

    def generate_pil_colors(self, labels: List[int]):
        return [self.colors[object_class] for object_class in labels]

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
        plt.clf()
        plt.close()

    def export(self, img_set, file_idx: int):
        if not isinstance(img_set, list):
            img_set = [img_set]

        fig, axs = plt.subplots(ncols=len(img_set), squeeze=False)

        for i, img in enumerate(img_set):
            img = img.detach()
            img = F.to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        plt.savefig(self.results_dir / Path('batch_' +
                    str(file_idx) + '.png'), dpi=400)
        plt.clf()
        plt.close()

    @torch.no_grad()
    def test_model(
        self,
        dataloader: torch.utils.data.DataLoader,
        results_dir: Path,
        no_visual: bool = True,
        no_save: bool = False,
        alpha: float = 0.5
    ):

        self.results_dir = results_dir
        for idx, batch in enumerate(dataloader):

            # This pre-processes raw batch
            batch = convert_image_dtype(
                batch, dtype=torch.float32).to(self.device)

            # Feed input to model
            outputs = self.model(batch)

            # Filter predictions with respect to to given thresholds
            boolean_masks = [
                out['masks'][out['scores'] >
                             self.conf_threshold] > self.prob_threshold
                for out in outputs
            ]

            # Get predicted labels
            labels = [
                out['labels'][out['scores'] > self.conf_threshold]
                for out in outputs
            ]

            colors = self.generate_pil_colors(
                list(labels[0].detach().cpu().numpy()))
            
            # If no objects are detected, skip
            if len(colors) > 0:
                
                # Draw predicted masks on image
                vis_result = [
                    draw_segmentation_masks(torch.tensor(
                        img * 255).type(torch.ByteTensor), mask.squeeze(1), colors=colors, alpha=alpha)
                    for img, mask in zip(batch, boolean_masks)
                ]

                if not no_visual:
                    # Visualize result
                    self.show(img_set=vis_result)

                if not no_save:
                    # Export result
                    self.export(img_set=vis_result, file_idx=idx)


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
