import argparse
import logging as log
import os
import sys

import cv2

from inference import Network

def load_data():
    def __init__(self, img_ids, img_dir, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id + '.png'))
        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']

        img = img.astype('float32') / 255

        img = img.transpose(2, 1, 0)

        return img, {'img_id': img_id}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./model/model.xml', help='model name')
    parser.add_argument('--origin_image_path', default='./data/test/')
    parser.add_argument('--test_image_path', default='./data/ray_test/', help='predict images dir path')
    parser.add_argument('--mask_path', default='./data/mask/', help='masks dir')
    parser.add_argument('--result_txt', default='./data/result/', help='')
    parser.add_argument('--outputs', default='./data/outputs/', help='outputs dir')
    parser.add_argument('--fc', default='./data/Fitting_Circle/', help='')
    parser.add_argument('--image_style', default='fluorescence', help='fluorescence, light')

    args = parser.parse_args()

    return args


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    model_xml = args.model
    TARGET_DEVICE = 'CPU'
    Net = Network()
    [batch_size, n_channels, height, width], exec_net, input_blob, out_blob = Net.load_model(model_xml, TARGET_DEVICE,
                                                                                             1, 1,
                                                                                             0)[1:5]
    print([batch_size, n_channels, height, width])
    print(exec_net, input_blob, out_blob)
    input_data, label_data, img_indicies = load_data()


if __name__ == '__main__':
    args = parse_args()
    main()
