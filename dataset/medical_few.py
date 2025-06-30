import os
import torch
from numpy.ma.core import masked
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import pandas as pd
import numpy as np

CLASS_NAMES = ['Brain', 'Liver', 'Retina_RESC', 'Retina_OCT2017', 'Chest', 'Histopathology']
CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}

# mask标签只有异常样本有，且是属于'Brain':3, 'Liver':2, 'Retina_RESC':1，这三类的才有mask标签


class MedDataset(Dataset):
    def __init__(self,
                 dataset_path='/data/',
                 class_name='Brain',
                 resize=240,
                 shot = 4,
                 iterate = -1
                 ):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        # 返回的shot的张量为（shot，C,H,W）
        assert shot>0, 'shot number : {}, should be positive integer'.format(shot)


        self.dataset_path = os.path.join(dataset_path, f'{class_name}_AD')
        self.resize = resize
        self.shot = shot
        self.iterate = iterate
        self.class_name = class_name
        self.seg_flag = CLASS_INDEX[class_name]

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder(self.seg_flag)

        # 图像变换管道-->针对图像
        # 图像重塑为 （240，240）【采用双三次插值的方法（上采样）】
        # 最后将图像转换为张量的形式

        self.transform_x = transforms.Compose([
            transforms.Resize((resize,resize), Image.BICUBIC),
            transforms.ToTensor(),
            ])

        # 图像变换管道-->针对掩码图像
        # 图像重塑为 （240，240）【采用最近邻插值的方法（上采样）】
        # 最后将图像转换为张量的形式
        self.transform_mask = transforms.Compose([
            transforms.Resize((resize,resize), Image.NEAREST),
            transforms.ToTensor()
            ])


        self.fewshot_norm_img = self.get_few_normal()
        self.fewshot_abnorm_img, self.fewshot_abnorm_mask = self.get_few_abnormal()
        
            

    def __getitem__(self, idx):
        # 图像路径x,标签y，mask图像级标签
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x).convert('RGB')
        x_img = self.transform_x(x)

        # 不考虑分割任务的情况
        if self.seg_flag < 0:
            return x_img, y, torch.zeros([1, self.resize, self.resize])

        # mask为空的情况下，mask也返回为0矩阵，y=0代表正常样本
        if mask is None:
            mask = torch.zeros([1, self.resize, self.resize])
            y = 0

        # 存在掩码则返回掩码，代表是异常样本
        else:
            mask = Image.open(mask).convert('L')
            mask = self.transform_mask(mask)
            y = 1
        return x_img, y, mask

    def __len__(self):
        return len(self.x)


    def load_dataset_folder(self, seg_flag):
        x, y, mask = [], [], []

        normal_img_dir = os.path.join(self.dataset_path, 'test', 'good', 'img')
        img_fpath_list = sorted([os.path.join(normal_img_dir, f) for f in os.listdir(normal_img_dir)])
        x.extend(img_fpath_list)
        y.extend([0] * len(img_fpath_list))
        mask.extend([None] * len(img_fpath_list))


        abnormal_img_dir = os.path.join(self.dataset_path, 'test', 'Ungood', 'img')
        img_fpath_list = sorted([os.path.join(abnormal_img_dir, f) for f in os.listdir(abnormal_img_dir)])
        x.extend(img_fpath_list)
        y.extend([1] * len(img_fpath_list))

        if self.seg_flag > 0:
            gt_fpath_list = [f.replace('img', 'anomaly_mask') for f in img_fpath_list]
            mask.extend(gt_fpath_list)
        else:
            mask.extend([None] * len(img_fpath_list))

        assert len(x) == len(y), 'number of x and y should be same'
        return list(x), list(y), list(mask)



    # 从数据集中获取少量正常样本图像。
    def get_few_normal(self):
        """
        获取少数正常图像样本。

        本函数根据设定的shot数量，从验证集中的正常图像中随机选择或按照预定义的选择顺序，
        选取指定数量的正常图像作为支持集样本。主要步骤包括：
        1. 确定图像目录并获取正常图像名称列表。
        2. 根据iterate参数和shot数量选择图像。
        3. 加载并预处理选中的图像。

        Returns:
            fewshot_img (torch.Tensor): 少数正常图像样本的张量，形状为(shot, C, H, W)。
        """
        # 初始化图像路径列表
        x = []
        # 构建正常图像目录路径
        img_dir = os.path.join(self.dataset_path, 'valid', 'good', 'img')
        # 获取正常图像文件名列表
        normal_names = os.listdir(img_dir)

        # 选择图像
        if self.iterate < 0:
            # 如果iterate小于0，随机选择shot数量的图像
            random_choice = random.sample(normal_names, self.shot)
        else:
            # 如果iterate大于等于0，按照预定义的选择顺序选择图像
            random_choice = []
            # 打开预定义的选择顺序文件
            with open(f'./dataset/fewshot_seed/{self.class_name}/{self.shot}-shot.txt', 'r', encoding='utf-8') as infile:
                # 逐行读取文件内容
                for line in infile:
                    # 去除首尾换行符，并按空格划分
                    data_line = line.strip("\n").split()
                    # 找到对应iterate的选择顺序
                    if data_line[0] == f'n-{self.iterate}:':
                        # 获取选中的图像文件名
                        random_choice = data_line[1:]
                        break

        # 根据选中的图像文件名构建完整的图像路径
        for f in random_choice:
            if f.endswith('.png') or f.endswith('.jpeg'):
                x.append(os.path.join(img_dir, f))

        # 初始化少数样本图像列表
        fewshot_img = []
        # 加载并预处理选中的图像
        for idx in range(self.shot):
            # x 保存的图像的路径列表，img取出每一个路径
            image = x[idx]
            # 打开图像并转换为RGB格式
            image = Image.open(image).convert('RGB')
            # 对图像进行变换
            image = self.transform_x(image)
            # 将处理后的图像添加到列表中
            # 并对处理后的图像添加一个维度（1，C,H,W）
            fewshot_img.append(image.unsqueeze(0))

        # 将列表中的图像张量合并为一个张量
        # 将这些少样本呢变成一个批次（shot,C,H,W）
        fewshot_img = torch.cat(fewshot_img)
        # 返回少数样本图像张量
        return fewshot_img

# 得到少量异常图像样本
    def get_few_abnormal(self):
        # x是存储图像路径的列表
        x = []
        # y是存储掩码路径的列表
        y = []

        # 设置异常图像及其标签路径
        img_dir = os.path.join(self.dataset_path, 'valid', 'Ungood', 'img')
        mask_dir = os.path.join(self.dataset_path, 'valid', 'Ungood', 'anomaly_mask')

        abnormal_names = os.listdir(img_dir)

        # 图像的选择规则，可以随机选择，也可以按预定义顺序选择
        # 用random_choice来存储图像路径列表
        # 最后把random_choice的路径列表赋给x和y
        # select images
        if self.iterate < 0:
            # 如果iterate小于0则按照随机选择数据
            random_choice = random.sample(abnormal_names, self.shot)
        else:
            # 如果大于0，则按照预定义顺序选择
            random_choice = []
            with open(f'./dataset/fewshot_seed/{self.class_name}/{self.shot}-shot.txt', 'r', encoding='utf-8') as infile:
                # 遍历txt文件内容，每行读取去除换行符，并按空格划分
                for line in infile:
                    data_line = line.strip("\n").split()  # 去除首尾换行符，并按空格划分
                    if data_line[0] == f'a-{self.iterate}:':
                        random_choice = data_line[1:]
                        break
        # 把刚才的路径名称random_choice的路径列表，赋给x和y
        for f in random_choice:
            if f.endswith('.png') or f.endswith('.jpeg'):
                x.append(os.path.join(img_dir, f))
                y.append(os.path.join(mask_dir, f))
        # fewshot_img来存储少量样本异常图像的张量
        fewshot_img = []
        # feshot_mask来存储小量样本异常图像的掩码张量（标签）
        fewshot_mask = []

        for idx in range(self.shot):
            # 获取路径
            image = x[idx]

            # 获取图像，转换为RGB
            image = Image.open(image).convert('RGB')

            # 这里对图像做了一系列变换，上面有定义这个管道，针对图像的变化
            image = self.transform_x(image)
            # 将处理后的图像添加一个维度（1，C,H,W）
            fewshot_img.append(image.unsqueeze(0))
            # 如果类别的编码大于0，说明这个类别有mask标签，则对y赋值
            if CLASS_INDEX[self.class_name] > 0:
                # image先获取路径
                image = y[idx]
                # 打开图片，并对图片进行变换为灰度图
                image = Image.open(image).convert('L')

                # 上面有对掩码图像变换的定义
                image = self.transform_mask(image)
                fewshot_mask.append(image.unsqueeze(0))

        fewshot_img = torch.cat(fewshot_img)

        # 如果没有掩码图像，那么这属于<0的那几类，只返回异常图像即可
        if len(fewshot_mask) == 0:
            return fewshot_img, None

        # 如果有，则返回异常图像和异常图像的标签
        else:
            fewshot_mask = torch.cat(fewshot_mask)
            return fewshot_img, fewshot_mask

