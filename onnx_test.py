"""
Unet++ pytorch model 2 onnx model
time:2021/8/12
version:v1
author:huangxiaohuang
"""

import os
import cv2
import onnxruntime
import torch
torch.cuda.current_device()
from albumentations import Compose
from albumentations.augmentations import transforms
from torch.utils.data import DataLoader

import L1_archs_cut


class test_Dataset(torch.utils.data.Dataset):
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


def pth_2onnx():
    """
    pytorch 模型转换为onnx模型
    :return:
    """
    torch_model = torch.load('./model/bf_t7/bright_field_model.pth')

    model = L1_archs_cut.NestedUNet(num_classes=1, input_channels=3, deep_supervision=True)
    model.load_state_dict(torch_model)
    batch_size = 1  # 批处理大小
    input_shape = (3, 1920, 1088)  # 输入数据

    # set the model to inference mode
    model.eval()
    print(model)
    x = torch.randn(batch_size, *input_shape)  # 生成张量
    export_onnx_file = "./model/bf_t7/bright_field_model.onnx"  # 目的ONNX文件名
    torch.onnx.export(model,
                      x,
                      export_onnx_file,
                      # 注意这个地方版本选择为11
                      opset_version=11,
                     )


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def image_test():
    """
    实际测试onnx模型效果
    :return:
    """
    onnx_path = './model/fluorescence_model.onnx'
    image_path = './data/ray_test/'

    test_transform = Compose([
        transforms.Normalize(),
    ])

    test_dataset = test_Dataset(
        img_ids='0',
        img_dir=image_path,
        num_classes=1,
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )

    for input, meta in test_loader:
        print('input_shape', input.shape)
        ort_session = onnxruntime.InferenceSession(onnx_path)

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
        print('ort_inputs_len', len(ort_inputs))
        print('ort_inputs', ort_inputs)
        ort_outs = ort_session.run(None, ort_inputs)
        print('ort_outs_type', type(ort_outs))
        print('ort_outs', ort_outs)
        img_out = ort_outs[0]
        print('111', img_out)
        img_out = torch.from_numpy(img_out)

        img_out = torch.sigmoid(img_out).cpu().numpy()

        print('img_out_shape', img_out.shape)
        print('img_out_type', type(img_out))
        print('img_out', img_out)
        img_out = img_out.transpose(0, 1, 3, 2)
        num_classes = 1
        for i in range(len(img_out)):
            cv2.imwrite(os.path.join('./', meta['img_id'][i].split('.')[0] + '.png'),
                        (img_out[i, num_classes - 1] * 255).astype('uint8'))


if __name__ == '__main__':
    pth_2onnx()
    # image_test()
"""
input_shape torch.Size([1, 3, 1920, 1088])
ort_inputs 1
img_out (1, 1, 1920, 1088)
"""