# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []


    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None

        flow = None
        if len(self.flow_list) != 0 and self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        elif len(self.flow_list) != 0:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        if flow is not None:
            flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        if len(self.flow_list) != 0:
            flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        elif flow is not None:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        else:
            return img1, img2

        return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='../sintel', dstype='clean', train=True):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            # if (not scene.startswith('ambush')) and train:
            #     continue
            # if scene.startswith('ambush') and not train:
            #     continue
            # print("scene:", scene)
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id
            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='../FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='../FlyingThings3D', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
      

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='../KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='../HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1


class nuScenes(FlowDataset):
    def __init__(self, aug_params=None, partition='front', root='../nuscenes/samples'):
        super(nuScenes, self).__init__(aug_params, sparse=False)

        front = sorted(glob(os.path.join(root, 'CAM_FRONT/*.jpg')))
        front_left = sorted(glob(os.path.join(root, 'CAM_FRONT_LEFT/*.jpg')))
        back_left = sorted(glob(os.path.join(root, 'CAM_BACK_LEFT/*.jpg')))
        back = sorted(glob(os.path.join(root, 'CAM_BACK/*.jpg')))
        back_right = sorted(glob(os.path.join(root, 'CAM_BACK_RIGHT/*.jpg')))
        front_right = sorted(glob(os.path.join(root, 'CAM_FRONT_RIGHT/*.jpg')))
        self.is_test = True

        print(len(front), len(front_left), len(front_right), len(back), len(back_left), len(back_right))

        for i in range(len(front) - 1):

            if partition == 'front':
                frame_id = front[i].split('/')[-1]
                self.image_list += [[front[i], front[i + 1]]]
            elif partition == 'front_left':
                frame_id = front_left[i].split('/')[-1]
                self.image_list += [[front_left[i], front_left[i + 1]]]
            elif partition == 'back_left':
                frame_id = back_left[i].split('/')[-1]
                self.image_list += [[back_left[i], back_left[i + 1]]]
            elif partition == 'back':
                frame_id = back[i].split('/')[-1]
                self.image_list += [[back[i], back[i + 1]]]
            elif partition == 'back_right':
                frame_id = back_right[i].split('/')[-1]
                self.image_list += [[back_right[i], back_right[i + 1]]]
            elif partition == 'front_right':
                frame_id = front_right[i].split('/')[-1]
                self.image_list += [[front_right[i], front_right[i + 1]]]
            self.extra_info += [[frame_id]]


# -----------------------------
# Add this class alongside the other datasets (e.g., after HD1K / before nuScenes)
# -----------------------------
class JHMDB(FlowDataset):
    """
    JHMDB loader for your folder layout:

      root/
        Frames/<action>/<video>/*.png
        FlowBrox04/<action>/<video>/*.jpg   (optional "flow", commonly encoded)

    If you DON'T want to use Brox flow as supervision, set use_flow=False and the dataset
    will return (img1, img2) like other flow-less datasets.

    If you DO want to use Brox flow, set use_flow=True. For .jpg flows, we assume a common
    encoding: u in R channel, v in G channel, decoded as (val - 128) / flow_img_scale.
    If your encoding differs, adjust `_read_flow_rg_jpg()`.
    """
    def __init__(
        self,
        aug_params=None,
        root="../JHMDB",
        use_flow=False,
        frame_ext="png",
        flow_ext="jpg",
        flow_mode="auto",         # "auto" or "rg_jpg" or "none"
        flow_img_scale=20.0       # decode scale for jpg flow
    ):
        super(JHMDB, self).__init__(aug_params, sparse=False)

        self.use_flow = use_flow
        self.flow_mode = flow_mode
        self.flow_img_scale = float(flow_img_scale)

        frames_root = osp.join(root, "Frames")
        flow_root   = osp.join(root, "FlowBrox04")

        if not osp.isdir(frames_root):
            raise FileNotFoundError(f"JHMDB Frames not found: {frames_root}")

        actions = sorted([d for d in os.listdir(frames_root) if osp.isdir(osp.join(frames_root, d))])

        for action in actions:
            action_dir = osp.join(frames_root, action)
            videos = sorted([d for d in os.listdir(action_dir) if osp.isdir(osp.join(action_dir, d))])

            for vid in videos:
                vid_dir = osp.join(action_dir, vid)
                frames = sorted(glob(osp.join(vid_dir, f"*.{frame_ext}")))
                if len(frames) < 2:
                    continue

                flow_vid_dir = osp.join(flow_root, action, vid)

                for i in range(len(frames) - 1):
                    img1_path, img2_path = frames[i], frames[i + 1]
                    stem = osp.splitext(osp.basename(img1_path))[0]  # e.g., "00001"

                    # If using flow supervision, only keep pairs where the flow exists
                    if self.use_flow:
                        flow_path = osp.join(flow_vid_dir, f"{stem}.{flow_ext}")
                        if not osp.isfile(flow_path):
                            # skip if flow file missing
                            continue
                        self.flow_list.append(flow_path)

                    self.image_list.append([img1_path, img2_path])
                    self.extra_info.append((action, vid, i))

    def _read_flow_rg_jpg(self, flow_path: str) -> np.ndarray:
        """Decode flow from a single JPG: u=R, v=G, flow=(val-128)/scale."""
        flow_img = frame_utils.read_gen(flow_path)
        flow_img = np.array(flow_img).astype(np.float32)

        if flow_img.ndim == 2:
            # grayscale: can't reliably recover (u,v); raise to avoid silent bugs
            raise ValueError(f"Expected RGB flow jpg but got grayscale: {flow_path}")

        # Use R,G channels as (u,v)
        u = flow_img[..., 0]
        v = flow_img[..., 1]
        u = (u - 128.0) / self.flow_img_scale
        v = (v - 128.0) / self.flow_img_scale
        return np.stack([u, v], axis=-1).astype(np.float32)

    def __getitem__(self, index):
        # copy FlowDataset logic but with custom flow decoding for JHMDB jpgs
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        flow = None

        # --- flow ---
        if self.use_flow and len(self.flow_list) != 0:
            flow_path = self.flow_list[index]

            if self.flow_mode == "none":
                flow = None
            elif self.flow_mode == "rg_jpg":
                flow = self._read_flow_rg_jpg(flow_path)
            else:
                # auto: if jpg -> rg_jpg decode, else try frame_utils.read_gen
                if str(flow_path).lower().endswith((".jpg", ".jpeg")):
                    flow = self._read_flow_rg_jpg(flow_path)
                else:
                    flow = frame_utils.read_gen(flow_path)
                    flow = np.array(flow).astype(np.float32)

        # --- images ---
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale -> 3ch
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        # --- augment ---
        if self.augmentor is not None:
            # your FlowAugmentor expects flow (can be None only if it supports it)
            if flow is None:
                # If you want to train without flow, keep use_flow=False so FlowDataset returns (img1,img2)
                img1, img2, flow = self.augmentor(img1, img2, flow)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        # --- to torch ---
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        if flow is None:
            return img1, img2

        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        return img1, img2, flow, valid.float()


class VirtualKITTI2(FlowDataset):
    def __init__(
        self,
        aug_params=None,
        root="../VKITTI2",
        scenes=("Scene01", "Scene02", "Scene06", "Scene18", "Scene20"),
        camera="Camera_0",
        include_forward=True,
        include_backward=True,
        variations=None,   # None => all subfolders under each scene
    ):
        super(VirtualKITTI2, self).__init__(aug_params, sparse=True)
        self.vkitti2 = True  # <<< tells FlowDataset to use read_vkitti2_flow

        # auto-handle extracted folder nesting: root/<vkitti_2.*>/Scene01/...
        base = root
        if not osp.isdir(osp.join(base, scenes[0])) and osp.isdir(base):
            for d in sorted(os.listdir(base)):
                cand = osp.join(base, d)
                if osp.isdir(cand) and osp.isdir(osp.join(cand, scenes[0])):
                    base = cand
                    break

        for scene in scenes:
            scene_dir = osp.join(base, scene)
            if not osp.isdir(scene_dir):
                continue

            if variations is None:
                types = sorted([d for d in os.listdir(scene_dir) if osp.isdir(osp.join(scene_dir, d))])
            else:
                types = list(variations)

            for v in types:
                type_dir = osp.join(scene_dir, v)
                if not osp.isdir(type_dir):
                    continue

                # rgb can be jpg (official), but keep jpg/png robust
                imgs = sorted(glob(osp.join(type_dir, "frames", "rgb", camera, "*.jpg")))
                if len(imgs) == 0:
                    imgs = sorted(glob(osp.join(type_dir, "frames", "rgb", camera, "*.png")))

                flows_fwd = sorted(glob(osp.join(type_dir, "frames", "forwardFlow", camera, "*.png")))
                flows_bwd = sorted(glob(osp.join(type_dir, "frames", "backwardFlow", camera, "*.png")))

                if len(imgs) < 2:
                    continue

                # forward: (t -> t+1)
                if include_forward and len(flows_fwd) == len(imgs) - 1:
                    for i in range(len(imgs) - 1):
                        self.image_list += [[imgs[i], imgs[i + 1]]]
                        self.flow_list += [flows_fwd[i]]
                        self.extra_info += [(scene, v, camera, "fwd", i)]

                # backward: (t+1 -> t)
                if include_backward and len(flows_bwd) == len(imgs) - 1:
                    for i in range(len(imgs) - 1):
                        self.image_list += [[imgs[i + 1], imgs[i]]]
                        self.flow_list += [flows_bwd[i]]
                        self.extra_info += [(scene, v, camera, "bwd", i)]



def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """

    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')

    elif args.stage == 'jhmdb':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        train_dataset = JHMDB(
            aug_params=aug_params,
            root=getattr(args, "jhmdb_root", "../JHMDB"),
            use_flow=getattr(args, "jhmdb_use_flow", False),      # set True if you want to use FlowBrox04 as flow GT
            flow_mode=getattr(args, "jhmdb_flow_mode", "auto"),   # "auto" or "rg_jpg"
            flow_img_scale=getattr(args, "jhmdb_flow_scale", 20.0),
            frame_ext=getattr(args, "jhmdb_frame_ext", "png"),
            flow_ext=getattr(args, "jhmdb_flow_ext", "jpg"),
        )

    elif args.stage in ('vkitti2', 'vkitti'):
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True}
        train_dataset = VirtualKITTI2(
            aug_params=aug_params,
            root=getattr(args, "vkitti2_root", "../VKITTI2"),
            camera=getattr(args, "vkitti2_camera", "Camera_0"),
            include_forward=True,
            include_backward=True,
            variations=getattr(args, "vkitti2_variations", None),  # None => all
        )


    torch.backends.cudnn.deterministic = True
    random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    np.random.seed(1234)

    def seed_worker(worker_id):
      seeds = [1, 2, 3, 4]
      worker_seed = seeds[worker_id]
      np.random.seed(seeds[worker_id])
      random.seed(seeds[worker_id])

    g = torch.Generator()
    g.manual_seed(0)


    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, drop_last=True, num_workers=2,
        worker_init_fn=seed_worker, generator=g)


    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

