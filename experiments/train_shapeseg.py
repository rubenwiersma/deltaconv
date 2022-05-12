import os, time, argparse
import os.path as osp
from progressbar import progressbar

import torch
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.transforms import Compose, GenerateMeshNormals
from torch_geometric.loader import DataLoader

from datasets import ShapeSeg
import deltaconv.transforms as T
from deltaconv.models import DeltaNetSegmentation

from utils import calc_loss


def train(args, writer):

    # Data preparation
    # ----------------

    # Download shapeseg.zip from https://surfdrive.surf.nl/files/index.php/s/L68uSYpHtfO6dLa
    # and unzip in experiments/data.
    # The folder structure in experiments/data should look like:
    # -------------------
    # v ShapeSeg
    #   v raw
    #     | shapeseg.zip
    # -------------------
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/ShapeSeg')

    # Apply pre-transformations: normalize, get mesh normals, and sample points on the mesh.
    pre_transform = Compose((
        T.NormalizeArea(),
        T.NormalizeAxes(),
        GenerateMeshNormals(),
        T.SamplePoints(args.num_points * args.sampling_margin, include_normals=True, include_labels=True),
        T.GeodesicFPS(args.num_points)
    ))

    # Transformations during training: random scale, rotation, and translation.
    transform = Compose((
        T.RandomScale((0.8, 1.2)),
        T.RandomRotate(360, axis=2),
        T.RandomTranslateGlobal(0.1),
        T.RandomNormals(0.01)
    ))

    # Load datasets.
    train_dataset = ShapeSeg(path, True, transform=transform, pre_transform=pre_transform)
    
    # Split the training set into a train/validation set used for early stopping.
    num_samples = len(train_dataset)
    num_train = int(num_samples * 0.9)
    num_validation = num_samples - num_train
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [num_train, num_validation], generator=torch.Generator().manual_seed(args.seed))

    # Load the separate test dataset.
    test_dataset = ShapeSeg(path, False, pre_transform=pre_transform)

    # And setup DataLoaders for each dataset.
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)


    # Model and optimization
    # ----------------------

    # Create the model.
    model = DeltaNetSegmentation(
        in_channels=3,                          # XYZ coordinates as input
        num_classes=8,                          # There are eight segmentation classes
        conv_channels=[128]*8,                  # We use 8 convolution layers, each with 128 channels
        mlp_depth=1,                            # Each convolution uses MLPs with only one layer (i.e., perceptrons)
        embedding_size=512,                     # Embed the features in 512 dimensions after convolutions
        num_neighbors=args.k,                   # The number of neighbors is given as an argument
        grad_regularizer=args.grad_regularizer, # The regularizer value is given as an argument 
        grad_kernel_width=args.grad_kernel,     # The kernel width is given as an argument 
    ).to(args.device)

    if not args.evaluating:
        # Setup optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


        # Train the model
        # ---------------

        best_validation = 0
        best_validation_test_score = 0
        for epoch in progressbar(range(1, args.epochs + 1)):
            train_epoch(epoch, model, args.device, optimizer, train_loader, writer)
            validation_accuracy = evaluate(model, args.device, validation_loader)
            writer.add_scalar('validation accuracy', validation_accuracy, epoch)
            test_accuracy = evaluate(model, args.device, test_loader)
            writer.add_scalar('test accuracy', test_accuracy, epoch)

            if validation_accuracy > best_validation:
                best_validation = validation_accuracy
                best_validation_test_score = test_accuracy
                torch.save(model.state_dict(), osp.join(args.checkpoint_dir, 'best.pt'))
            scheduler.step()
    else:
        model.load_state_dict(torch.load(args.checkpoint))
        best_validation_test_score = evaluate(model, args.device, test_loader)

    print("Test accuracy: {}".format(best_validation_test_score))


def train_epoch(epoch, model, device, optimizer, loader, writer):
    """Train the model for one iteration on each item in the loader."""
    model.train()
    running_loss = 0.0

    for i, data in enumerate(loader):
        optimizer.zero_grad()
        out = model(data.to(device))
        loss = calc_loss(out, data.y, smoothing=False)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 50 == 49:
            writer.add_scalar('training loss',
                            running_loss / 50,
                            epoch * len(loader) + i)

            running_loss = 0.0
    model.train()


def evaluate(model, device, loader):
    """Evaluate the model for on each item in the loader."""
    model.eval()
    correct = 0
    total_num = 0
    for data in loader:
        pred = model(data.to(device)).max(1)[1]
        correct += pred.eq(data.y).sum().item()
        total_num += data.y.size(0)
    eval_acc = correct / total_num
    return eval_acc


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='DeltaNet Segmentation')
    # Optimization hyperparameters.
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch (default: 8)')
    parser.add_argument('--epochs', type=int, default=50, metavar='num_epochs',
                        help='Number of episode to train (default: 50)')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Number of points to use (default: 1024)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='Learning rate (default: 0.005)')

    # DeltaConv hyperparameters. 
    parser.add_argument('--k', type=int, default=20, metavar='K',
                        help='Number of nearest neighbors to use (default: 20)')
    parser.add_argument('--grad_kernel', type=float, default=1, metavar='h',
                        help='Kernel size for WLS, as a factor of the average edge length (default: 1)')
    parser.add_argument('--grad_regularizer', type=float, default=0.001, metavar='lambda',
                        help='Regularizer lambda to use for WLS (default: 0.001)')

    # Dataset generation arguments. 
    parser.add_argument('--sampling_margin', type=int, default=8, metavar='sampling_margin',
                        help='The number of points to sample before using FPS to downsample (default: 8)')

    # Logging and debugging.
    parser.add_argument('--logdir', type=str, default='', metavar='logdir',
                        help='Root directory of log files. Log is stored in LOGDIR/runs/EXPERIMENT_NAME/TIME. (default: FILE_PATH)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Evaluation.
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to the checkpoint to evaluate. The script will only evaluate if a path is given.')

    args = parser.parse_args()

    # If a checkpoint is given, evaluate the model rather than training.
    args.evaluating = args.checkpoint != ''

    # Determine the device to run the experiment
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Name the experiment, used to store logs and checkpoints.
    args.experiment_name = 'shapeseg'
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