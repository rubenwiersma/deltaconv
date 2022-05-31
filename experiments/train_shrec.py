import os, time, argparse
import os.path as osp
from progressbar import progressbar

import torch
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.transforms import Compose, SamplePoints
from torch_geometric.loader import DataLoader

from datasets import SHREC
import deltaconv.transforms as T
from deltaconv.models import DeltaNetClassification

from utils import calc_loss
import sklearn.metrics as metrics
import numpy as np


def train(args, writer):

    # Data preparation
    # ----------------

    # Path to the dataset folder
    # The dataset will be downloaded if it is not yet available in the given folder.
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/shrec')

    # Pre-transformations: normalize and sample points on the mesh.
    pre_transform = Compose((
        T.NormalizeScale(),
        SamplePoints(args.num_points * args.sampling_margin, include_normals=True),
        T.GeodesicFPS(args.num_points)
    ))

    # Transformations during training: random rotation and translation.
    transform = Compose((
        T.RandomRotate(360, 0),
        T.RandomRotate(360, 1),
        T.RandomRotate(360, 2),
        T.RandomTranslateGlobal(0.1),
    ))

    # Load datasets.
    train_dataset = SHREC(path, train=True, transform=transform, pre_transform=pre_transform)
    test_dataset = SHREC(path, train=False, pre_transform=pre_transform)

    # And setup DataLoaders for each dataset.
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)


    # Model and optimization
    # ----------------------

    # Create the model.
    model = DeltaNetClassification(
        in_channels=3,
        num_classes=30,
        conv_channels=[32]*4,
        num_neighbors=args.k,
        grad_regularizer=args.grad_regularizer,
        grad_kernel_width=args.grad_kernel
    ).to(args.device)

    if not args.evaluating:
        # Setup optimizer and scheduler
        optimizer = torch.optim.SGD(model.parameters(), lr=100 * args.lr, momentum=args.momentum, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr)


        # Train the model
        # ---------------

        for epoch in progressbar(range(1, args.epochs + 1)):
            train_epoch(epoch, model, args.device, optimizer, train_loader, writer)
            test_acc = evaluate(model, args.device, test_loader)
            writer.add_scalar('test accuracy', test_acc, epoch)
            scheduler.step()
        torch.save(model.state_dict(), osp.join(args.checkpoint_dir, 'last.pt'))
    else:
        model.load_state_dict(torch.load(args.checkpoint))
        test_acc = evaluate(model, args.device, test_loader)

    print("Test accuracy: {}".format(test_acc))


def train_epoch(epoch, model, device, optimizer, loader, writer):
    """Train the model for one iteration on each item in the loader."""
    model.train()

    total_loss = 0
    running_loss = 0.0
    train_pred = []
    train_true = []
    for i, data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = calc_loss(out, data.y, smoothing=True)
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
        running_loss += loss.item()
        train_pred.append(out.max(dim=1)[1].detach().cpu().numpy())
        train_true.append(data.y.cpu().numpy())
        if i % 50 == 49:
            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 50,
                            epoch * len(loader) + i)

            running_loss = 0.0
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    train_acc = metrics.accuracy_score(train_true, train_pred)
    writer.add_scalar('training accuracy', train_acc, epoch)


def evaluate(model, device, loader):
    """Evaluate the model for on each item in the loader."""
    model.eval()

    correct = 0
    eval_pred = []
    eval_true = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        eval_pred.append(pred.detach().cpu().numpy())
        eval_true.append(data.y.cpu().numpy())
    eval_true = np.concatenate(eval_true)
    eval_pred = np.concatenate(eval_pred)
    eval_acc = metrics.accuracy_score(eval_true, eval_pred)
    return eval_acc


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='DeltaNet SHREC classification')

    # Optimization hyperparameters.
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Size of batch (default: 16)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of episode to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')

    # DeltaConv hyperparameters.
    parser.add_argument('--k', type=int, default=20,
                        help='Number of nearest neighbors to use (default: 20)')
    parser.add_argument('--grad_regularizer', type=float, default=0.001, metavar='lambda',
                        help='Regularizer lambda to use for WLS (default: 0.001)')
    parser.add_argument('--grad_kernel', type=float, default=1,
                        help='Kernel size for weighted least squares gradient (default: 1)')

    # Dataset generation arguments.
    parser.add_argument('--sampling_margin', type=int, default=8,
                        help='The number of points to sample before using FPS to downsample (default: 8)')
    parser.add_argument('--num_points', type=int, default=2048, metavar='N',
                        help='Number of points to use (default: 2048)')

    # Logging and debugging.
    parser.add_argument('--logdir', type=str, default='',
                        help='Root directory of log files. Log is stored in LOGDIR/runs/EXPERIMENT_NAME/TIME. (default: FILE_PATH)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    
    # Evaluation.
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to the checkpoint to evaluate. The script will only evaluate if a path is given.')

    args = parser.parse_args()

    # If a checkpoint is given, evaluate the model rather than training.
    args.evaluating = args.checkpoint != ''

    # Determine the device to run the experiment on.
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Name the experiment, used to store logs and checkpoints.
    args.experiment_name = 'shrec'
    run_time = time.strftime("%d%b%y_%H_%M", time.localtime(time.time()))

    writer = None
    if not args.evaluating:
        # Set log directory and create TensorBoard writer in log directory.
        if args.logdir == '':
            args.logdir = osp.dirname(osp.realpath(__file__))
        args.logdir = osp.join(args.logdir, 'runs', args.experiment_name, run_time)
        writer = SummaryWriter(args.logdir)

        # Create directory to store checkpoints. 
        args.checkpoint_dir = osp.join(args.logdir, 'checkpoints')
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        # Write experimental details to log directory.
        experiment_details = args.experiment_name + '\n--\nSettings:\n--\n'
        for arg in vars(args):
            experiment_details += '{}: {}\n'.format(arg, getattr(args, arg))
        with open(os.path.join(args.logdir, 'settings.txt'), 'w') as f:
            f.write(experiment_details)

        # And show experiment details in console.
        print(experiment_details)
        print('---')
        print('Training...')
    else:
        print('Evaluating {}...'.format(args.experiment_name))

    # Start training process
    torch.manual_seed(args.seed)
    train(args, writer)