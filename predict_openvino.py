"""
Unet++ 模型推理使用openvino
time：2021/08/16
version：v1
author：huangxiaohuang
"""

import argparse
import datetime
import logging as log
import os
import sys
from glob import glob
from openvino.inference_engine import IECore
import cv2
import torch
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./model/fluorescence_model.xml', help='model name')
    parser.add_argument('--origin_image_path', default='./data/ray_test/')
    parser.add_argument('--test_image_path', default='./data/ray_test/', help='predict images dir path')
    parser.add_argument('--mask_path', default='./data/mask/', help='masks dir')
    parser.add_argument('--result', default='./data/result/', help='')

    args = parser.parse_args()

    return args


class load_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id))
        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']

        img = img.astype('float32') / 255

        img = img.transpose(2, 1, 0)

        return img, {'img_id': img_id}


def load_data(test_image_path):
    # step6 准备输入
    image_path = test_image_path

    test_transform = Compose([
        transforms.Normalize(),
    ])

    img_ids = glob(os.path.join(image_path, '*'))

    test_img_ids = [os.path.basename(p) for p in img_ids]

    test_dataset = load_Dataset(
        img_ids=test_img_ids,
        img_dir=image_path,
        num_classes=1,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )

    return test_loader


def load_model(model_path):
    log.basicConfig(format='[%(levelname)s] %(message)s', level=log.INFO, stream=sys.stdout)

    # step1 初始化推理引擎
    log.info('Create Inference Engine')
    ie = IECore()

    # step2 读取openvino模型中间件
    log.info(f'Reading the network:{args.model}')
    net = ie.read_network(model=model_path)

    # step3 配置输入和输出
    log.info('Configuring input and output blobs')

    # 获取输入输出的blobs
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    # step4 加载模型至设备上
    log.info('Loading the model to the plugin')
    exec_net = ie.load_network(network=net, device_name='CPU')

    return net, input_blob, out_blob, exec_net


def main():
    # 加载模型
    net, input_blob, out_blob, exec_net = load_model(args.model)
    # 加载数据
    test_loader = load_data(args.origin_image_path)

    for input, meta in test_loader:
        log.info('Starting inference in synchronous mode')

        # 进行推理
        res = exec_net.infer(inputs={input_blob: input})

        # 从字典中获取相应的结果
        res = res[out_blob]

        img_out = torch.from_numpy(res)
        img_out = torch.sigmoid(img_out).cpu().numpy()

        # 改变输出的形状
        img_out = img_out.transpose(0, 1, 3, 2)

        for i in range(len(img_out)):
            cv2.imwrite(os.path.join(args.result, meta['img_id'][i].split('.')[0] + '.png'),
                        (img_out[i, 0] * 255).astype('uint8'))


if __name__ == '__main__':
    args = parse_args()
    t1 = datetime.datetime.now()
    print(t1)
    main()
    t2 = datetime.datetime.now()
    print(t2)
    print('yongshi:', t2 - t1)
