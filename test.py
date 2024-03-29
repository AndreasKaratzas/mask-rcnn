
import os
import json
import torch
import datetime
import warnings
import argparse

from pathlib import Path
from torch.utils.data import DataLoader

from lib.visual import Visual
from lib.utils import colorstr
from lib.nvidia import cuda_check
from lib.model import configure_model
from lib.dataloader import TestDataset
from lib.transformation import get_transform

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of Mask R-CNN.')
    parser.add_argument('--model-checkpoint', type=str, help='Path to pretrained model.')
    parser.add_argument('--dataset', type=str, help='Path to dataset.')
    parser.add_argument('--output', default='demo',
                        type=str, help='Path to output the test results (default: demo).')
    parser.add_argument('--img-size', default=800, type=int, help='Image size.')
    parser.add_argument('--batch-size', default=1, type=int, help='Batch size.')
    parser.add_argument('--num-classes', default=2, type=int, help='Number of classes in dataset.')
    parser.add_argument('--backbone', default='resnet50', help='Backbone CNN for Mask R-CNN.')
    parser.add_argument('--num-workers', default=1, type=int, metavar='N', help='Number of data loading workers ('
                                                                                'default: 1).')
    parser.add_argument('--conf-threshold', default=0.50, type=float, help='Confidence threshold in prediction.')
    parser.add_argument('--visualize', default=0.40, type=float, help='Visualize a percentage of predictions.')
    parser.add_argument('--anchor-sizes', default=[32, 64, 128, 256, 512], nargs='+', type=int, help='Anchor sizes.')
    parser.add_argument('--aspect-ratios', default=[0.5, 1.0, 2.0], nargs='+', type=int, help='Anchor ratios.')
    parser.add_argument('--no-visual', action='store_true',
                        help='Disable visualization software in test mode.')
    parser.add_argument('--no-save', action='store_true',
                        help='Disable results export software.')
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    if not Path(args.model_checkpoint).is_file():
        raise ValueError(f"Path to pretrained model weights is invalid. Value parsed {args.model_checkpoint}.")
    if not Path(args.dataset).is_dir():
        raise ValueError(f"Path to dataset is invalid. Value parsed {args.dataset}.")

    test_results_dir = None
    if not args.no_save:
        datetime_tag = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        test_results_dir = Path(args.model_checkpoint).parent.absolute(
        ).parent.absolute() / Path(args.output) / datetime_tag
        test_results_dir.mkdir(parents=True, exist_ok=True)

    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(
        f"Device utilized: {colorstr(options=['red', 'underline'], string_args=list([device]))}.\n")

    if device == torch.device('cuda'):
        args.n_devices, cuda_arch = cuda_check()
        print(
            f"Found NVIDIA GPU of "
            f"{colorstr(options=['cyan'], string_args=list([cuda_arch]))} "
            f"Architecture.")

    with open(Path(args.model_checkpoint).parent.absolute().parent.absolute().joinpath('CONFIG.json')) as f:
        data = json.load(f)

        args.backbone = data['model']['backbone']
        args.anchor_sizes = data['model']['anchors']
        args.aspect_ratios = data['model']['ratios']
        args.epochs = data['model']['epochs']

        args.num_classes = data['dataset']['classes']
        args.img_size = data['dataset']['img_size']

        # Typecast list of `str` to list of `float`
        args.anchor_sizes = [float(anchor) for anchor in args.anchor_sizes]
        args.aspect_ratios = [float(ratio) for ratio in args.aspect_ratios]

    # custom model init
    model = configure_model(
        backbone_name=args.backbone,
        anchor_sizes=args.anchor_sizes,
        aspect_ratios=args.aspect_ratios,
        num_classes=args.num_classes,
        min_size=args.img_size
    )

    # test dataset
    test_data = TestDataset(
        root_dir=os.path.join(args.dataset, "test"),
        transforms=get_transform(
            transform_class="test")
    )

    # test dataloader
    dataloader_test = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )

    # load model to device
    model = model.to(device)

    # load checkpoint
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded model checkpoint at {args.model_checkpoint} successfully")

    # test model
    visualize = Visual(model=model, root_dir=args.dataset,
                       device=device, conf_threshold=args.conf_threshold, 
                       num_classes=args.num_classes)
    visualize.test_model(dataloader=dataloader_test, results_dir=test_results_dir,
                         no_visual=args.no_visual, no_save=args.no_save)
