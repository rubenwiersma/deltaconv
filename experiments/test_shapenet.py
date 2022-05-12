
import os, time, argparse
import os.path as osp
from progressbar import progressbar

import torch
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_mean

from datasets import ShapeNet
import deltaconv.transforms as T
from deltaconv.models import DeltaNetSegmentation

from utils import calc_loss, calc_shape_IoU
import sklearn.metrics as metrics
import numpy as np

from .train_shapenet import shapenet_model


def test(args):

    # Data preparation
    # ----------------

    # Path to the dataset folder
    # The dataset will be downloaded if it is not yet available in the given folder.
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/ShapeNet')

    # Pre-transformations: normalize and sample points with FPS.
    pre_transform = Compose((
        T.NormalizeScale(),
        T.GeodesicFPS(args.num_points) # It is assumed that GeodesicFPS is used in the multiscale architecture
    ))

    # Transformations during training: random scale and translation.
    transform = Compose((
        T.RandomScale((2/3, 3/2)),
        T.RandomTranslateGlobal(0.1)
    ))
    
    # Load dataset.
    test_dataset = ShapeNet(path, categories=args.class_choice, split='test', pre_transform=pre_transform, transform=transform)

    # And setup DataLoader.
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    # The number of classes is either the number of parts in one shape category
    # or the total number of classes for all shape categories.
    num_classes = test_dataset.num_classes if args.class_choice is None else len(test_dataset.seg_classes[args.class_choice])


    # Model and optimization
    # ----------------------

    # Create the model.
    model = shapenet_model(args, num_classes)


    # Load and evaluate the model
    # ---------------------------
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict)
    evaluate(model, args.device, test_loader, args)


def evaluate(model, device, loader, args):
    model.eval()

    test_pred_seg_acc = None
    test_pred_seg = []

    test_true_seg = []
    test_label_seg = []
    for i in progressbar(range(args.num_votes)):
        for data in loader:
            data = data.to(device)
            with torch.no_grad():
                pred = model(data)
            test_pred_seg.append(pred.detach().cpu().numpy().reshape(data.num_graphs, -1, pred.size(1)))
            
            if i == 0:
                test_true_seg.append(data.y.cpu().numpy().reshape(data.num_graphs, -1))
                test_label_seg.append(data.category.max(dim=1)[1].cpu().numpy())
        if test_pred_seg_acc is None:
            test_pred_seg_acc = np.concatenate(test_pred_seg, axis=0)
        else:
            test_pred_seg_acc += np.concatenate(test_pred_seg, axis=0)
        test_pred_seg = []

    # Aggregate probabilities of N runs and select most probable class
    test_pred_seg = np.argmax(test_pred_seg_acc, axis=2)
    test_true_seg = np.concatenate(test_true_seg, axis=0)

    test_acc = metrics.accuracy_score(test_true_seg.flatten(), test_pred_seg.flatten())
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_seg.flatten(), test_pred_seg.flatten())

    test_label_seg = np.concatenate(test_label_seg)
    test_ious = calc_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)

    print('test mean iou: ', np.mean(test_ious))
    print('test accuracy: ', test_acc)
    print('test avg class accuracy', avg_per_class_acc)

    mean_iou_per_class = scatter_mean(torch.Tensor(test_ious), torch.from_numpy(test_label_seg)).numpy()
    for i, (key, _) in enumerate(loader.dataset.category_ids.items()):
        print('iou {}: '.format(key), mean_iou_per_class[i])
    return test_ious


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='DeltaNet ShapeNet segmentation')

    # Checkpoint path to evaluate. 
    parser.add_argument('--checkpoint', type=str, default='', metavar='checkpoint',
                        help='Path to checkpoint you want to evaluate.')

    # Evaluation hyperparameters.
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Size of batch (default: 16)')
    parser.add_argument('--num_votes', type=int, default=10,
                        help='Number of votes used for evaluation (default: 10)')

    # DeltaConv hyperparameters.
    parser.add_argument('--k', type=int, default=30, 
                        help='Number of nearest neighbors to use (default: 30)')
    parser.add_argument('--grad_regularizer', type=float, default=0.001, metavar='lambda',
                        help='Regularizer lambda to use for WLS (default: 0.001)')
    parser.add_argument('--grad_kernel', type=float, default=1, 
                        help='Kernel size for weighted least squares gradient (default: 1)')

    # Dataset arguments.
    parser.add_argument('--class_choice', type=str, default=None, 
                        choices=['Airplane', 'Bag', 'Cap', 'Car', 'Chair',
                                 'Earphone', 'Guitar', 'Knife', 'Lamp', 'Laptop', 
                                 'Motorbike', 'Mug', 'Pistol', 'Rocket', 'Skateboard', 'Table'])
    parser.add_argument('--num_points', type=int, default=2048, metavar='N',
                        help='Number of points to use (default: 2048)')

    # Logging and debugging.
    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed (default: 1)')

    args = parser.parse_args()

    # Determine the device to run the experiment on.
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Start training process
    torch.manual_seed(args.seed)
    test(args)