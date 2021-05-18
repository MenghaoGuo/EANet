import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import DataLoader

import settings
from network import EANet

logger = settings.logger


def ensure_dir(dir_path): 
    if not osp.isdir (dir_path):
        os.makedirs(dir_path)


def fetch(image_path, label_path=None):
    with open(image_path, 'rb') as fp:
        image = Image.open(fp).convert('RGB')
    image = torch.FloatTensor(np.asarray(image)) / 255
    image = (image - settings.MEAN) / settings.STD
    image = image.permute(2, 0, 1).unsqueeze(dim=0)

    if label_path is not None:
        with open(label_path, 'rb') as fp:
            label = Image.open(fp).convert('P')
        label = torch.FloatTensor(np.asarray(label))
        label = label.unsqueeze(dim=0).unsqueeze(dim=1)
    else:
        label = None

    return image, label


def pad_inf(image, label=None):
    h, w = image.size()[-2:] 
    stride = settings.STRIDE
    pad_h = (stride + 1 - h % stride) % stride
    pad_w = (stride + 1 - w % stride) % stride
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0.)
        if label is not None:
            label = F.pad(label, (0, pad_w, 0, pad_h), mode='constant', 
                          value=settings.IGNORE_LABEL)
    return image, label


class BaseDataset(data.Dataset):
    def __init__(self, data_root, split):
        self.data_root = data_root

        file_list = osp.join('datalist', split + '.txt')
        file_list = tuple(open(file_list, 'r'))
        file_list = [id_.rstrip() for id_ in file_list]
        self.files = file_list

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_id = self.files[idx]
        return self._get_item(image_id)

    def _get_item(self, idx):
        raise NotImplementedError


class TestDataset(BaseDataset):
    def __init__(self, data_root, split='test'):
        super(TestDataset, self).__init__(data_root, split)

    def _get_item(self, image_id):
        image_path = osp.join(self.data_root, image_id + '.jpg')
        image, _ = fetch(image_path)

        return image[0], image_id


class TestSession(object):
    def __init__(self, dt_split):
        self.log_dir = settings.LOG_DIR
        self.model_dir = settings.MODEL_DIR

        self.net = EANet(settings.N_CLASSES, settings.N_LAYERS).cuda()
        self.net = DataParallel(self.net)
        dataset = TestDataset(data_root=settings.TEST_DATA_ROOT, split=dt_split)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, 
                                     num_workers=16, drop_last=False)
        self.hist = 0 

    def load_checkpoints(self, name):
        ckp_path = name
        try:
            obj = torch.load(ckp_path, 
                             map_location=lambda storage, loc: storage.cuda())
            logger.info('Load checkpoint %s.' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!' % ckp_path)
            return

        self.net.module.load_state_dict(obj['net'])

    def inf_batch(self, image):
        image = image.cuda()
        with torch.no_grad():
            logit = self.net(image)

        return logit 


def trans_scale(image, h, w):
    image = F.interpolate(image, size=(h, w), mode='bilinear', 
                          align_corners=True)
    return image


def get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def test_main(ckp_name='final.pth'):
    sess = TestSession('test')
    sess.load_checkpoints(ckp_name)
    dt_iter = sess.dataloader
    sess.net.eval()
    
    save_dir = osp.join(settings.TEST_SAVE_DIR, settings.EXP_NAME)
    logger.info('Set Test save dir, %s' % save_dir)
    ensure_dir(save_dir)

    for _, [image, image_id] in enumerate(dt_iter):
        _, _, h, w = image.size()
        logits = np.zeros((1, settings.N_CLASSES,h, w), np.float32)
        logits = torch.Tensor (logits)
        test_scale = list (settings.TEST_SCALES)

        # orig
        for scale in test_scale:     
            scale_h = (int) (scale * h)
            scale_w = (int) (scale * w)
            
            scale_image = trans_scale(image, scale_h, scale_w)
            scale_image, _ = pad_inf(scale_image)
            
            scale_logit = sess.inf_batch(scale_image)

            scale_logit = scale_logit[:, :, 0:scale_h, 0:scale_w]
            scale_logit = F.interpolate(scale_logit, size=[h, w], mode='bilinear', align_corners=True)
            logits += scale_logit.cpu() 

        # flip
        for scale in test_scale : 
            scale_h = (int) (scale * h)
            scale_w = (int) (scale * w)

            flip_image = torch.flip(image, [3])
            flip_image = trans_scale(flip_image, scale_h, scale_w)
            flip_image, _ = pad_inf(flip_image)
            
            flip_logit = sess.inf_batch(flip_image)

            flip_logit = flip_logit[:, :, 0:scale_h, 0:scale_w]
            flip_logit = F.interpolate(flip_logit, size=[h, w], mode='bilinear', align_corners=True)
            logit = torch.flip(flip_logit, [3])
            logits += logit.cpu() 



        pred = logits.max(dim=1)[1]
        
        # save results
        pred_arr = np.array(pred.cpu().data)
        pred_image = Image.fromarray(np.uint8(pred_arr[0]))

        palette = get_palette(256) 
        pred_image.putpalette(palette)
        
        name = image_id[0].split('.')[0]
        save_path = osp.join(save_dir, f'{name}.png')
        pred_image.save(save_path)


if __name__ == '__main__':
    load_dir = osp.join(settings.MODEL_DIR, settings.EXP_NAME)
    load_name = osp.join(load_dir, 'final.pth')
    test_main(load_name)
