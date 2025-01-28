import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image, ImageOps

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import torchvision.transforms as T




DEVICE = 'cuda'

def load_image(imfile, transform):
    image = Image.open(imfile)
    image = transform(image)
    img = np.array(image).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


# class_boundary = list(np.arange(0, 16, 2))
# class_boundary.append(400)
# class_boundary = list(np.arange(0, 400, 400//10))
# class_boundary.append(400)

def viz(args, img1, img2, flo, gt_flo, path, _id):
    img = img1[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    boundary = max(img.shape[0], img.shape[1])
    gt_flo = gt_flo[0].permute(1,2,0).cpu().numpy()
    gt_flo = np.where(gt_flo > boundary, boundary, gt_flo)
    flo = flo[0].permute(1,2,0).cpu().numpy()
    flo = np.where(flo > boundary, boundary, flo)

    
    # map flow to rgb image
    # gt_flo = flow_viz.flow_to_image(gt_flo)
    # flo = flow_viz.flow_to_image(flo)

    try:
        os.mkdir(args.output_path)
    except Exception as e:
        print(f"Couldn't create directory {args.output_path}: {e}")
        return 
    
    if len(path):
        try:
            os.mkdir(os.path.join(args.output_path, path))
        except Exception as e:
            print(f"Couldn't create directory {args.output_path}: {e}")
            return 
    
    # mag = np.sqrt(np.sum(flo**2, axis=2)) 
    # _class = np.zeros(mag.shape)
    # for i in range(len(class_boundary) - 1):
    #     _class += np.where((class_boundary[i] < mag) & (mag < class_boundary[i + 1]), len(class_boundary) - i, 0)

    # _class = _class / (len(class_boundary) - 1) * 255

    # flox_rgb = Image.fromarray(_class.astype('uint8'), 'RGB')
    # flox_gray = ImageOps.grayscale(flox_rgb)    
    # flox_gray = Image.fromarray(_class.astype('uint8'), 'L')    
    output_path = os.path.join(args.output_path, path)

    # flox_rgb = Image.fromarray(gt_flo.astype('uint8'), 'RGB')
    # flox_rgb.save(output_path + '/diff_flow_' + _id + '.png')
    # flox_rgb = Image.fromarray(flo.astype('uint8'), 'RGB')
    # flox_rgb.save(output_path + '/predicted_flow_' + _id + '.png')

    flox_rgb = Image.fromarray(img.astype('uint8'), 'RGB')
    flox_rgb.save(output_path + '/' + 'attacked_img' + _id + '.png')
    flox_rgb = Image.fromarray(img2.astype('uint8'), 'RGB')
    flox_rgb.save(output_path + '/' + 'noise' + _id + '.png')

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()

def fgsm_attack(image, epsilon, data_grad):
    # sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 255)
    return perturbed_image

def demo(args):
    transform = T.Resize((240, 427))

    torch.cuda.empty_cache()

    model = torch.nn.DataParallel(RAFT(args))
    if not args.raft:
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    # with torch.no_grad():
    paths = []
    for entry in os.scandir(args.path):
        if entry.is_dir():
            new_path = args.path + '/' + entry.name
            paths.append(new_path)
    if len(paths) == 0:
        paths.append(args.path)

    # cwd = os.getcwd()
    # args.output_path = os.path.join(cwd, args.output_path)


    for path in paths:
        _id = 0
        images = glob.glob(os.path.join(path, '*.png')) + \
                    glob.glob(os.path.join(path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1, transform)
            image2 = load_image(imfile2, transform)
            print(torch.max(image1), torch.min(image1))
            print(image1.shape)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            
            # start attack
            if args.attack_type != 'None':
                image1.requires_grad = True # for attack
            
            torch.cuda.empty_cache()

            ori = image1.data.clone().detach()
            flow_gt = padder.unpad(flow_up[0]).clone().detach()

            if args.attack_type != 'None':
                if args.attack_type == 'FGSM':
                    epsilon = args.epsilon
                    pgd_iters = 1
                else:
                    epsilon = 2.5 * args.epsilon / args.iters
                    pgd_iters = args.iters
                
                shape = image1.shape
                delta = (np.random.rand(np.product(shape)).reshape(shape) - 0.5) * 2 
                image1.data = ori + torch.from_numpy(delta).type(torch.float).cuda()
                image1.data = torch.clamp(image1.data, 0.0, 255.0)
                flow_low, flow_pr = model(image1, image2, iters=20, test_mode=True)
            
                for iter in range(pgd_iters):
                    flow = padder.unpad(flow_pr[0])
                    epe = torch.sum((flow - flow_gt.cuda())**2, dim=0).sqrt().view(-1)
                    model.zero_grad()
                    image1.requires_grad = True
                    epe.mean().backward(retain_graph=True)
                    data_grad = image1.grad.data
                    args.channel = int(args.channel)
                    if args.channel == -1:
                        image1.data = fgsm_attack(image1, epsilon, data_grad)
                    else:
                        image1.data[:, args.channel, :, :] = fgsm_attack(image1, epsilon, data_grad)[:, args.channel, :, :]
                    if args.attack_type == 'PGD':
                        offset = image1.data - ori
                        image1.data = ori + torch.clamp(offset, -args.epsilon, args.epsilon)
                flow_low, flow_pr = model(image1, image2, iters=20, test_mode=True)
            folder_name = path[len(args.path):]
            viz(args, image1.detach(), (image1.data - ori).detach(), (flow_pr - flow_up).detach(), flow_pr.detach(), folder_name, str(_id))
            _id += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--output_path', help="output viz")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--raft', help="checkpoint from the RAFT paper?", type=bool, default=True)
    parser.add_argument('--fcbam', help='Add CBAM after the feature network?', type=bool, default=False)
    parser.add_argument('--ccbam', help='Add CBAM after the context network?', type=bool, default=False)
    parser.add_argument('--deform', help='Add deformable convolution?', type=bool, default=False)
    parser.add_argument('--attack_type', help='Attack type options: None, FGSM, PGD', type=str, default='FGSM')
    parser.add_argument('--epsilon', help='epsilon?', type=int, default=10.0)
    parser.add_argument('--channel', help='Color channel options: 0, 1, 2, -1 (all)', type=int, default=-1)  
    parser.add_argument('--iters', help='Number of iters for PGD?', type=int, default=50) 
    args = parser.parse_args()

    demo(args)
