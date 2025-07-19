from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from raft import RAFT
import evaluate
import core.datasets as datasets
from scipy import ndimage
from scipy.interpolate import interp2d
from utils import flow_viz
from PIL import Image, ImageOps


from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000
# class_boundary = list(np.arange(1, 400, 400//10))
class_boundary = list(np.arange(0, 16, 2))
class_boundary.append(400)
print(class_boundary)



def viz(img1, img2, flo, gt_flo, path = '', _id = '1'):
    print(img1[0].shape, img2[0].shape, flo[0].shape, gt_flo[0].shape)
    img = img1[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    gt_flo = gt_flo[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()

    gt_flo = flow_viz.flow_to_image(gt_flo)
    flo = flow_viz.flow_to_image(flo)



    # entries = os.listdir(path)
    # print(entries)

    # try:
    #     os.mkdir(args.output_path)
    # except Exception as e:
    #     pass
    
    # if len(path):
    #     try:
    #         os.mkdir(os.path.join(args.output_path, path))
    #     except Exception as e:
    #         pass
    
    output_path = path

    flox_rgb = Image.fromarray(gt_flo.astype('uint8'), 'RGB')
    flox_rgb.save(output_path + 'gt_flow_' + _id + '.png')
    flox_rgb = Image.fromarray(flo.astype('uint8'), 'RGB')
    flox_rgb.save(output_path + 'composed_flow_' + _id + '.png')

    flox_rgb = Image.fromarray(img.astype('uint8'), 'RGB')
    flox_rgb.save(output_path + 'image1' + _id + '.png')
    flox_rgb = Image.fromarray(img2.astype('uint8'), 'RGB')
    flox_rgb.save(output_path + 'image2' + _id + '.png')



def compose_flow_batch(flow1, flow2):
    """
    Compose optical flow from image a to c using optical flows from a to b and b to c.
    Handles batches of flows.
    
    Parameters:
    flow1 : torch.Tensor
        Optical flow from image a to b, shape (batch_size, 2, h, w)
    flow2 : torch.Tensor
        Optical flow from image b to c, shape (batch_size, 2, h, w)
    
    Returns:
    composed : torch.Tensor
        Optical flow from image a to c, shape (batch_size, 2, h, w)
    """
    batch_size, _, h, w = flow1.shape
    

    N, _, H, W = flow1.shape

    # # --- 1. Create a base grid of pixel coordinates ---
    # # This grid represents the original pixel locations 'p'.
    # x_coords = torch.linspace(0, W - 1, W, device=flow1.device)
    # y_coords = torch.linspace(0, H - 1, H, device=flow1.device)
    # grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    # base_grid_pixels = torch.stack((grid_x, grid_y), dim=2) # Shape: (H, W, 2)
    
    # # Expand the grid to match the batch size N without copying data.
    # batch_base_grid = base_grid_pixels.unsqueeze(0).expand(N, -1, -1, -1) # Shape: (N, H, W, 2)

    # # --- 2. Calculate target sampling coordinates in pixel space ---
    # # The target coordinates are p' = p + flow1(p).
    # # We need to reshape flow1 to match the grid for addition.
    # flow1_for_grid = flow1.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
    # sampling_grid_pixels = batch_base_grid + flow1_for_grid # Shape: (N, H, W, 2)

    # # --- 3. Normalize the sampling grid for grid_sample ---
    # # grid_sample requires coordinates in the range [-1, 1].
    # norm_factor = torch.tensor([W - 1, H - 1], dtype=torch.float32, device=flow1.device)
    # normalized_sampling_grid = 2.0 * (sampling_grid_pixels / norm_factor) - 1.0

    # --- 4. Warp the second flow field using grid_sample ---
    # # grid_sample is designed for batches, so this works directly.
    # warped_flow2_tensor = F.grid_sample(
    #     flow2,
    #     normalized_sampling_grid,
    #     mode='bilinear',
    #     padding_mode='zeros', # Use (0,0) flow for out-of-bounds samples
    #     align_corners=True
    # )

    # --- 6. Add the first flow and the warped second flow ---
    # composed = flow1 + warped_flow2_tensor
    grid = self._create_normalized_grid(N, H, W)
        
    # Convert flow1 to sampling grid
    # We need to add flow1 to pixel coordinates, then normalize
    flow1_permuted = flow1.permute(0, 2, 3, 1)  # (N, 2, H, W) -> (N, H, W, 2)
    
    # Create pixel coordinate grid
    pixel_coords = self._create_pixel_grid(N, H, W)
    
    # Add flow1 to get new pixel locations
    new_pixel_coords = pixel_coords + flow1_permuted
    
    # Normalize to [-1, 1] for grid_sample
    new_pixel_coords[..., 0] = 2.0 * new_pixel_coords[..., 0] / (W - 1) - 1.0
    new_pixel_coords[..., 1] = 2.0 * new_pixel_coords[..., 1] / (H - 1) - 1.0

    warped_flow2 = F.grid_sample(
            flow2, 
            new_pixel_coords,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
    composed = flow1 + warped_flow2

    
    return composed


def composition_loss(flow_preds1, flow_preds2, flow_preds12, gamma):
    n_predictions = len(flow_preds1)    
    flow_loss = 0.0
    flow_composed_ls = []

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        # batch1, batch2 = flow_preds1[i].cpu().detach().numpy(), flow_preds2[i].cpu().detach().numpy()
        # flow_composed = np.zeros_like(batch1)

        batch1, batch2 = flow_preds1[i], flow_preds2[i]
        # flow_composed = torch.zeros_like(batch1)
        
        # for b in range(len(batch1)):
        #     flow_composed[b] = compose_flow_single(batch1[b], batch2[b])

        flow_composed = compose_flow_batch(batch1, batch2)

        # print(flow_composed.shape)

        # flow_composed = torch.from_numpy(flow_composed).to(dtype=flow_preds1[i].dtype, device=flow_preds1[i].device)
        
        flow_composed_ls.append(flow_composed)
        i_loss = (flow_composed - flow_preds12[i]).abs()
        flow_loss += i_weight * i_loss.mean()

    epe = torch.sum((flow_composed_ls[-2] - flow_preds12[-2])**2, dim=1).sqrt()
    epe = epe.view(-1)

    # cross_entropy = loss(class_gt, flow_preds[-1])

    metrics = {
        # 'loss': flow_loss.item(),
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics, flow_composed_ls[-2]



def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    # print(flow_gt.shape)
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    mag = torch.unsqueeze(mag, 1)
    mag = torch.tile(mag, (1, 2, 1, 1))
    # class_gt = torch.zeros_like(mag)
    # print(mag.shape, flow_gt.shape)
    # for i in range(len(class_boundary) - 1):
    #     class_gt += torch.where((class_boundary[i] < mag) & (mag < class_boundary[i + 1]), len(class_boundary) - i, 0)

    for i in range(n_predictions):
        # loss = nn.CrossEntropyLoss()
        # cross_entropy = loss(class_gt, flow_preds[i])
            
        i_weight = gamma**(n_predictions - i - 1)
        
        i_loss = (flow_preds[i] - flow_gt).abs()
        # i_loss = (flow_preds[i] - flow_gt)**2
        # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # i_loss = -cos(flow_preds[i], flow_gt)
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()


    epe = torch.sum((flow_preds[-2] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    # cross_entropy = loss(class_gt, flow_preds[-1])

    metrics = {
        # 'loss': flow_loss.item(),
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler, total_steps):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = total_steps
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0

    scaler = GradScaler(enabled=args.mixed_precision)

    if args.cont is not None:
        checkpoint = torch.load(args.cont)
        total_steps = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    logger = Logger(model, scheduler, total_steps)

    VAL_FREQ = 5000

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, image3, flow1, flow2, valid = [x.cuda() for x in data_blob]


            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
                image3 = (image3 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)


            if args.flow_composition:
                flow_predictions12 = model(image1, image2, iters=args.iters)  
                flow_predictions23 = model(image2, image3, iters=args.iters)     
                flow_predictions13 = model(image1, image3, iters=args.iters)  
                loss, metrics, composed = composition_loss(flow_predictions12, flow_predictions23, flow_predictions13, args.gamma)
                if i_batch == 1:
                    viz(image1, image3, composed.detach(), flow_predictions13[-2].detach())
                    exit()
            else:
                flow_predictions = model(image1, image2, iters=args.iters)            
                loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)
            
            if (total_steps + 1)  % SUM_FREQ == SUM_FREQ-1:
              torch.save({
              'step': total_steps,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'scheduler_state_dict': scheduler.state_dict(),
              'scaler_state_dict': scaler.state_dict(),
              }, 'checkpoints/last.pth')

            PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save({
                'step': total_steps,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                }, PATH)
                # torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module))

                logger.write_dict(results)
                
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
                # should_keep_training = False
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    # torch.save(model.state_dict(), PATH)
    torch.save({
                'step': total_steps,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                }, PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--cont', help="continue training from checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.000001) # 0.00002
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true', default=False)

    parser.add_argument('--fcbam', help='Add CBAM after the feature network?', type=bool, default=False)
    parser.add_argument('--ccbam', help='Add CBAM after the context network?', type=bool, default=False)
    parser.add_argument('--deform', help='Add deformable convolution?', type=bool, default=False)
    parser.add_argument('--flow_composition', type=bool, default=True)


    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)
    
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)
