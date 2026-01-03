"""
Script for training Regressor Model
"""

import argparse
import os
from datetime import datetime
import logging

from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset_multibin import Dataset
from models.orientation_dimension_resnet18 import ResNet18, orientationLoss
from config import load_config


# ---------------------- Setup & Utilities ---------------------- #

def setup_logging():
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO
    )


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    return device


def create_dataloader(kitti_dir, output_dimensions_path, batch_size, num_workers):
    dataset = Dataset(kitti_dir, output_dimensions_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def create_model(device, lr):
    base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model = ResNet18(model=base_model).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return model, optimizer


def create_scheduler(optimizer, step_size, gamma):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path, config):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': config
    }, save_path)
    logging.info(f"Checkpoint saved: {save_path}")


# ---------------------- Training Logic ---------------------- #

def train_one_epoch(model, dataloader, device, optimizer, loss_funcs, alpha, w, writer, global_step, lr):
    model.train()
    epoch_losses = {'total': 0, 'conf': 0, 'orient': 0, 'dim': 0}

    with tqdm(dataloader, unit='batch', desc="Training") as tepoch:
        for local_batch, local_labels in tepoch:
            # Move data to device
            local_batch = local_batch.float().to(device)
            truth_orient = local_labels['orientation'].float().to(device)
            truth_conf = local_labels['confidence'].float().to(device)
            truth_dim = local_labels['dimensions'].float().to(device)

            # Forward
            orient, conf, dim = model(local_batch)

            # Loss calculation
            orient_loss = loss_funcs['orient'](orient, truth_orient, truth_conf)
            dim_loss = loss_funcs['dim'](dim, truth_dim)
            conf_labels = torch.max(truth_conf, dim=1)[1]
            conf_loss = loss_funcs['conf'](conf, conf_labels)

            total_loss = alpha * dim_loss + conf_loss + w * orient_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Logging
            tepoch.set_postfix(loss=total_loss.item(), lr=lr)
            for k, v in zip(['total', 'conf', 'orient', 'dim'],
                            [total_loss.item(), conf_loss.item(), orient_loss.item(), dim_loss.item()]):
                epoch_losses[k] += v

            if writer and global_step % 100 == 0:
                writer.add_scalar('Loss/Total_Step', total_loss.item(), global_step)
                writer.add_scalar('Loss/Confidence_Step', conf_loss.item(), global_step)
                writer.add_scalar('Loss/Orientation_Step', orient_loss.item(), global_step)
                writer.add_scalar('Loss/Dimension_Step', dim_loss.item(), global_step)
                writer.add_scalar('Learning_Rate', lr, global_step)

            global_step += 1

    return epoch_losses, global_step


def train(cfg):
    setup_logging()
    device = get_device()

    config_data = load_config(cfg.config_path)
    kitti_dir = config_data['kitti_path']

    os.makedirs(cfg.output_folder, exist_ok=True)

    writer = None
    if cfg.use_tensorboard:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(cfg.log_dir, f'train_{timestamp}')
        writer = SummaryWriter(log_path)
        logging.info(f"Tensorboard logs at {log_path}")

    dataloader = create_dataloader(kitti_dir, cfg.output_dimensions_path, cfg.batch_size, cfg.num_workers)
    model, optimizer = create_model(device, cfg.lr)
    scheduler = create_scheduler(optimizer, cfg.scheduler_step_size, cfg.scheduler_gamma)

    loss_funcs = {
        'conf': nn.CrossEntropyLoss().to(device),
        'dim': nn.MSELoss().to(device),
        'orient': orientationLoss
    }

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        epoch_losses, global_step = train_one_epoch(
            model, dataloader, device, optimizer, loss_funcs,
            cfg.alpha, cfg.w, writer, global_step, optimizer.param_groups[0]['lr']
        )

        scheduler.step()

        # Epoch logging
        if writer:
            for k, v in epoch_losses.items():
                writer.add_scalar(f'Loss/{k.capitalize()}_Epoch', v / len(dataloader), epoch)
            writer.add_scalar('Learning_Rate_Epoch', optimizer.param_groups[0]['lr'], epoch)

        # Save checkpoint
        if epoch % cfg.save_epoch == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, epoch_losses['total'] / len(dataloader),
                            os.path.join(cfg.output_folder, f'multibin_epoch_{epoch}.pt'),
                            vars(cfg))

    if writer:
        writer.close()
        logging.info("Training completed.")


# ---------------------- CLI ---------------------- #

def parse_opt():
    parser = argparse.ArgumentParser(description='Regressor Model Training')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--w', type=float, default=0.4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--save_epoch', type=int, default=10)
    parser.add_argument('--config_path', type=str, default='config/default.yaml')
    parser.add_argument('--output_folder', type=str, default='weights')
    parser.add_argument('--no_tensorboard', action='store_true')
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--scheduler_step_size', type=int, default=10)
    parser.add_argument('--scheduler_gamma', type=float, default=0.2)
    parser.add_argument('--output_dimensions_path', type=str, default='calibration/class_dimensions.json')
    args = parser.parse_args()
    args.use_tensorboard = not args.no_tensorboard
    return args


if __name__ == '__main__':
    opt = parse_opt()
    train(opt)