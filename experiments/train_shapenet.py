import os, time, argparse
import os.path as osp
from progressbar import progressbar

import torch
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader

from datasets import ShapeNet
import deltaconv.transforms as T
from deltaconv.models import DeltaNetSegmentation

from utils import calc_loss, calc_shape_IoU
import sklearn.metrics as metrics
import numpy as np


def train(args, writer):

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
        T.RandomTranslateGlobal(0.2)
    ))
    
    # Load datasets.
    # We do not use a validation set for this task, but simply use the last epoch.
    train_dataset = ShapeNet(path, categories=args.class_choice, split='trainval', transform=transform, pre_transform=pre_transform)
    test_dataset = ShapeNet(path, categories=args.class_choice, split='test', pre_transform=pre_transform)

    # And setup DataLoaders for each dataset.
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    # The number of classes is either the number of parts in one shape category
    # or the total number of classes for all shape categories.
    num_classes = train_dataset.num_classes if args.class_choice is None else len(train_dataset.seg_classes[args.class_choice])

    # Model and optimization
    # ----------------------

    # Define the model
    model = shapenet_model(args, num_classes)

    # Setup optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=100 * args.lr, momentum=args.momentum, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr)

    # Train the model
    # ---------------

    for epoch in progressbar(range(1, args.epochs + 1)):
        train_epoch(epoch, model, args.device, optimizer, train_loader, writer, args)
        test_ious = evaluate(model, args.device, test_loader, args)
        writer.add_scalar('test mean iou', np.mean(test_ious), epoch)
        scheduler.step()
    torch.save(model.state_dict(), osp.join(args.checkpoint_dir, 'last.pt'))


def shapenet_model(args, num_classes):
    """ Define ShapeNet model in a separate function, so it can be reused by the test script. """
    return DeltaNetSegmentation(
        in_channels=3,                          # XYZ coordinates as input
        num_classes=num_classes,                # The number of classes is determined before.
        conv_channels=[64, 128, 256],           # We use 3 convolution layers.
        mlp_depth=2,                            # Each convolution uses MLPs with two layers.
        embedding_size=1024,                    # Embed the features in 1024 dimensions after convolutions.
        num_neighbors=args.k,                   # The number of neighbors is given as an argument.
        grad_regularizer=args.grad_regularizer, # The regularizer value is given as an argument.
        grad_kernel_width=args.grad_kernel,     # The kernel width is given as an argument. 
        categorical_vector=True                 # Use the shape category in the task-specific head.
    ).to(args.device)


def train_epoch(epoch, model, device, optimizer, loader, writer, args):
    """Train the model for one iteration on each item in the loader."""
    model.train()

    total_loss = 0
    running_loss = 0.0
    train_pred_seg = []
    train_true_seg = []
    train_label_seg = []
    for i, data in enumerate(loader):
        data = data.to(device)
        if args.class_choice is not None:
            labels = data.y - data.y.min()
        else:
            labels = data.y
        optimizer.zero_grad()
        out = model(data)
        loss = calc_loss(out, labels, smoothing=False)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
        running_loss += loss.item()
        pred = out.max(dim=1)[1].detach().cpu().numpy()
        true = labels.cpu().numpy()
        train_pred_seg.append(pred.reshape(data.num_graphs, -1))
        train_true_seg.append(true.reshape(data.num_graphs, -1))
        train_label_seg.append(data.category.max(dim=1)[1].cpu().numpy())
        if i % 50 == 49:
            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 50,
                            epoch * len(loader) + i)

            running_loss = 0.0
    train_true_seg = np.concatenate(train_true_seg, axis=0)
    train_pred_seg = np.concatenate(train_pred_seg, axis=0)
    train_acc = metrics.accuracy_score(train_true_seg.flatten(), train_pred_seg.flatten())
    avg_per_class_acc = metrics.balanced_accuracy_score(train_true_seg.flatten(), train_pred_seg.flatten())
    train_label_seg = np.concatenate(train_label_seg)
    train_ious = calc_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, args.class_choice)
    writer.add_scalar('training mean iou', np.mean(train_ious), epoch)
    writer.add_scalar('training accuracy', train_acc, epoch)
    writer.add_scalar('training avg class accuracy', avg_per_class_acc, epoch)


def evaluate(model, device, loader, args):
    """Evaluate the model for on each item in the loader."""
    model.eval()

    correct = 0
    eval_pred_seg = []
    eval_true_seg = []
    eval_label_seg = []
    for data in loader:
        data = data.to(device)
        if args.class_choice is not None:
            labels = data.y - data.y.min()
        else:
            labels = data.y
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(labels).sum().item()
        eval_pred_seg.append(pred.detach().cpu().numpy().reshape(data.num_graphs, -1))
        eval_true_seg.append((labels).cpu().numpy().reshape(data.num_graphs, -1))
        eval_label_seg.append(data.category.max(dim=1)[1].cpu().numpy())
    eval_true_seg = np.concatenate(eval_true_seg, axis=0)
    eval_pred_seg = np.concatenate(eval_pred_seg, axis=0)
    eval_label_seg = np.concatenate(eval_label_seg)
    eval_ious = calc_shape_IoU(eval_pred_seg, eval_true_seg, eval_label_seg, args.class_choice)
    return eval_ious


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='DeltaNet ShapeNet segmentation')
    
    # Optimization hyperparameters.
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Size of batch (default: 16)')
    parser.add_argument('--epochs', type=int, default=200, 
                        help='Number of episode to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='SGD momentum (default: 0.9)')

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
    parser.add_argument('--logdir', type=str, default='', 
                        help='Root directory of log files. Log is stored in LOGDIR/runs/EXPERIMENT_NAME/TIME. (default: FILE_PATH)')
    parser.add_argument('--checkpoint_dir', type=str, default='',
                        help='Directory to store checkpoints. Will default to the log directory if left empty.')
    args = parser.parse_args()

    # Determine the device to run the experiment on.
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Name the experiment, used to store logs and checkpoints.
    args.experiment_name = 'shapenet_{}/'.format(args.class_choice if args.class_choice else 'all')
    run_time = time.strftime("%d%b%y_%H_%M", time.localtime(time.time()))

    # Set log directory and create TensorBoard writer in log directory.
    if args.logdir == '':
        args.logdir = osp.dirname(osp.realpath(__file__))
    args.logdir = osp.join(args.logdir, 'runs', args.experiment_name, run_time)
    writer = SummaryWriter(args.logdir)

    # Create directory to store checkpoints. 
    args.checkpoint_dir = osp.join(args.logdir, 'checkpoints') if args.checkpoint_dir == '' else args.checkpoint_dir
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # Write experiment details to log directory
    experiment_details = args.experiment_name + '\n--\nSettings:\n--\n'
    for arg in vars(args):
        experiment_details += '{}: {}\n'.format(arg, getattr(args, arg))
    with open(os.path.join(args.logdir, 'settings.txt'), 'w') as f:
        f.write(experiment_details)

    print(experiment_details)
    print('---')
    print('Training...')

    # Start training process
    torch.manual_seed(args.seed)
    train(args, writer)