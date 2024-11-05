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



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


# class_boundary = list(np.arange(0, 16, 2))
# class_boundary.append(400)
# class_boundary = list(np.arange(0, 400, 400//10))
# class_boundary.append(400)

def viz(args, img1, img2, flo):
    img = img1[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    # gt_flo = gt_flo[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    # gt_flo = flow_viz.flow_to_image(gt_flo)
    flo = flow_viz.flow_to_image(flo)
    try:
        os.mkdir(args.output_path)
    except:
        pass
    
    # mag = np.sqrt(np.sum(flo**2, axis=2)) 
    # _class = np.zeros(mag.shape)
    # for i in range(len(class_boundary) - 1):
    #     _class += np.where((class_boundary[i] < mag) & (mag < class_boundary[i + 1]), len(class_boundary) - i, 0)

    # _class = _class / (len(class_boundary) - 1) * 255

    # flox_rgb = Image.fromarray(_class.astype('uint8'), 'RGB')
    # flox_gray = ImageOps.grayscale(flox_rgb)    
    # flox_gray = Image.fromarray(_class.astype('uint8'), 'L')    

    # flox_rgb = Image.fromarray(gt_flo.astype('uint8'), 'RGB')
    # flox_rgb.save(args.output_path + '/ground_truth_flow.png')
    flox_rgb = Image.fromarray(flo.astype('uint8'), 'RGB')
    flox_rgb.save(args.output_path + '/predicted_flow.png')

    flox_rgb = Image.fromarray(img.astype('uint8'), 'RGB')
    flox_rgb.save(args.output_path + '/' + 'img1.png')
    flox_rgb = Image.fromarray(img2.astype('uint8'), 'RGB')
    flox_rgb.save(args.output_path + '/' + 'img2.png')

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    if not args.raft:
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            print(torch.max(image1), torch.min(image1))

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            print(image1)
            viz(args, image1, image2, flow_up)


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
    args = parser.parse_args()

    demo(args)
