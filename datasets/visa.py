# from collections import defaultdict
# from pathlib import Path
# import random
# import torch
# import PIL
# import pandas as pd
# from torchvision.transforms.v2.functional import pil_to_tensor
# from .base import BaseDataset, DatasetSplit
# from torchvision import transforms as T
# from PIL import Image
#
#
# class VisADataset(torch.utils.data.Dataset):
#     CLASSNAMES = [
#         "candle",
#         "capsules",
#         "cashew",
#         "chewinggum",
#         "fryum",
#         "macaroni1",
#         "macaroni2",
#         "pcb1",
#         "pcb2",
#         "pcb3",
#         "pcb4",
#         "pipe_fryum",
#     ]
#
#     def __init__(self, source: str, classname: str, imagesize: int, shot: int = 4, iterate: int = 0, transform=None, target_transform=None):
#         super().__init__()
#         self.source = Path(source)
#         self.classname = classname  # 确保这里有赋值
#         self.imagesize = imagesize
#         self.shot = shot
#         self.iterate = iterate
#         self.transform = transform
#         self.target_transform = target_transform
#
#         # 加载 CSV 数据
#         csv_path = self.source / "split_csv" / "1cls.csv"
#         if not csv_path.exists():
#             raise FileNotFoundError(f"未找到 CSV 文件: {csv_path}")
#         self.csv_data = pd.read_csv(csv_path)
#
#         # 筛选当前类别的数据
#         self.class_data = self.csv_data[self.csv_data['object'] == self.classname].reset_index(drop=True)
#         print(f"类别 '{classname}' 加载的样本数量: {len(self.class_data)}")  # 添加调试信息
#
#         # 初始化 few-shot 相关属性
#         self.fewshot_abnorm_img = []
#         self.fewshot_abnorm_mask = []
#         self.fewshot_norm_img = []
#
#         # 加载 few-shot 样本
#         self._load_fewshot_samples()
#
#     def _load_fewshot_samples(self):
#         """加载 few-shot 样本"""
#         for idx, row in self.class_data.iterrows():
#             image_path = self.source / row['image']  # 确保这里是正确的路径
#             if not image_path.exists():
#                 print(f"警告: 找不到图像文件: {image_path}")
#                 continue
#
#             # 加载图像
#             img = Image.open(image_path).convert("RGB")
#             img = img.resize((self.imagesize, self.imagesize))  # 调整图像大小
#
#             if self.transform:
#                 img = self.transform(img)
#
#             # 加载掩码（如果存在）
#             mask_path = self.source / row['mask']  # 确保这里是正确的路径
#             if mask_path.exists():
#                 mask = Image.open(mask_path).convert("L")  # 转换为灰度图
#                 mask = mask.resize((self.imagesize, self.imagesize))  # 调整掩码大小
#                 if self.target_transform:
#                     mask = self.target_transform(mask)
#             else:
#                 mask = torch.zeros((1, self.imagesize, self.imagesize))  # 如果没有掩码，创建空掩码
#
#             # 根据样本的异常类型将图像和掩码添加到相应的列表中
#             if row['label'] == 'normal':
#                 self.fewshot_norm_img.append(img)
#                 self.fewshot_abnorm_img.append(torch.zeros_like(img))  # 正常样本的异常图像为零
#                 self.fewshot_abnorm_mask.append(mask)
#             else:
#                 self.fewshot_abnorm_img.append(img)
#                 self.fewshot_abnorm_mask.append(mask)
#
#         # 转换为张量并确保它们是正确的形状
#         if self.fewshot_norm_img:
#             self.fewshot_norm_img = torch.stack(self.fewshot_norm_img)
#         else:
#             self.fewshot_norm_img = torch.empty((0, 3, self.imagesize, self.imagesize))
#
#         if self.fewshot_abnorm_img:
#             self.fewshot_abnorm_img = torch.stack(self.fewshot_abnorm_img)
#             self.fewshot_abnorm_mask = torch.stack(self.fewshot_abnorm_mask)
#         else:
#             self.fewshot_abnorm_img = torch.empty((0, 3, self.imagesize, self.imagesize))
#             self.fewshot_abnorm_mask = torch.empty((0, 1, self.imagesize, self.imagesize))
#
#         # 打印形状以进行调试
#         print("Normal images shape:", self.fewshot_norm_img.shape if len(self.fewshot_norm_img) else "empty")
#         print("Abnormal images shape:", self.fewshot_abnorm_img.shape if len(self.fewshot_abnorm_img) else "empty")
#         print("Abnormal masks shape:", self.fewshot_abnorm_mask.shape if len(self.fewshot_abnorm_mask) else "empty")
#
#         if not (len(self.fewshot_norm_img) and len(self.fewshot_abnorm_img)):
#             raise RuntimeError(f"Could not load enough few-shot samples from {self.source}")
#
#         # 确保所有张量都是4D的 [B, C, H, W]
#         if len(self.fewshot_norm_img.shape) == 3:
#             self.fewshot_norm_img = self.fewshot_norm_img.unsqueeze(0)
#         if len(self.fewshot_abnorm_img.shape) == 3:
#             self.fewshot_abnorm_img = self.fewshot_abnorm_img.unsqueeze(0)
#         if len(self.fewshot_abnorm_mask.shape) == 3:
#             self.fewshot_abnorm_mask = self.fewshot_abnorm_mask.unsqueeze(0)
#
#     # def __getitem__(self, idx):
#     #     classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
#     #
#     #     # 加载图像
#     #     image = PIL.Image.open(image_path).convert("RGB")
#     #     original_img_width, original_img_height = image.size
#     #
#     #     # 应用图像变换
#     #     image = self.transform_img[0](image)
#     #
#     #     # 加载掩码（如果存在）
#     #     if mask_path is not None:
#     #         mask = PIL.Image.open(mask_path)
#     #         mask = (pil_to_tensor(mask) != 0).float()
#     #         mask = self.transform_mask[0](mask)
#     #     else:
#     #         mask = torch.zeros([1, self.resize, self.resize])
#     #
#     #     return {
#     #         "image": image,
#     #         "mask": mask,
#     #         "classname": classname,
#     #         "anomaly": anomaly,
#     #         "is_anomaly": int(anomaly != "normal"),
#     #         "image_name": Path(image_path).stem,
#     #         "image_path": str(image_path),
#     #         "mask_path": str(mask_path) if mask_path else "None",
#     #         "original_img_height": original_img_height,
#     #         "original_img_width": original_img_width,
#     #     }
#
#     def __getitem__(self, idx):
#         classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
#
#         # 加载图像
#         image = PIL.Image.open(image_path).convert("RGB")
#         original_img_width, original_img_height = image.size
#
#         # 应用图像变换
#         image = self.transform_img[0](image)
#
#         # 加载掩码（如果存在）
#         if mask_path is not None:
#             mask = PIL.Image.open(mask_path)
#             mask = (pil_to_tensor(mask) != 0).float()
#             mask = self.transform_mask[0](mask)
#         else:
#             mask = torch.zeros([1, self.imagesize, self.imagesize])  # 修正：使用 self.imagesize
#
#         return {
#             "image": image,
#             "mask": mask,
#             "classname": classname,
#             "anomaly": anomaly,
#             "is_anomaly": int(anomaly != "normal"),
#             "image_name": Path(image_path).stem,
#             "image_path": str(image_path),
#             "mask_path": str(mask_path) if mask_path else "None",
#             "original_img_height": original_img_height,
#             "original_img_width": original_img_width,
#         }
#
#     def get_image_data(self):
#         """获取图像数据路径"""
#         imgpaths_per_class = defaultdict(lambda: defaultdict(list))
#         maskpaths_per_class = defaultdict(lambda: defaultdict(list))
#
#         # 对于每个类别
#         for classname in self.classnames_to_use:
#             class_path = self.source / classname / "Data"
#
#             # 加载正常图像
#             normal_path = class_path / "Images" / "Normal"
#             if normal_path.exists():
#                 for img_path in normal_path.glob("*.JPG"):
#                     imgpaths_per_class[classname]["good"].append(img_path)
#                     maskpaths_per_class[classname]["good"].append(None)
#
#             # 加载异常图像和掩码
#             anomaly_img_path = class_path / "Images" / "Anomaly"
#             anomaly_mask_path = class_path / "Masks" / "Anomaly"
#             if anomaly_img_path.exists() and anomaly_mask_path.exists():
#                 for img_path in anomaly_img_path.glob("*.JPG"):
#                     mask_path = anomaly_mask_path / img_path.name
#                     if mask_path.exists():
#                         imgpaths_per_class[classname]["anomaly"].append(img_path)
#                         maskpaths_per_class[classname]["anomaly"].append(mask_path)
#
#         # 创建数据列表
#         data_to_iterate = []
#         for classname in sorted(imgpaths_per_class.keys()):
#             for anomaly in sorted(imgpaths_per_class[classname].keys()):
#                 for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
#                     data_tuple = [classname, anomaly, image_path]
#                     if anomaly != "good":
#                         data_tuple.append(maskpaths_per_class[classname][anomaly][i])
#                     else:
#                         data_tuple.append(None)
#                     data_to_iterate.append(data_tuple)
#
#         return dict(imgpaths_per_class), dict(maskpaths_per_class), data_to_iterate


from collections import defaultdict
from pathlib import Path
import random
import torch
import PIL
import pandas as pd
from torchvision.transforms.v2.functional import pil_to_tensor
from .base import BaseDataset, DatasetSplit
from torchvision import transforms as T
from PIL import Image


class VisADataset(torch.utils.data.Dataset):
    CLASSNAMES = [
        "candle",
        "capsules",
        "cashew",
        "chewinggum",
        "fryum",
        "macaroni1",
        "macaroni2",
        "pcb1",
        "pcb2",
        "pcb3",
        "pcb4",
        "pipe_fryum",
    ]

    # def __init__(self, source: str, classname: str, imagesize: int, shot: int = 4, iterate: int = 0, transform=None,
    #              target_transform=None):
    #     super().__init__()
    #     self.source = Path(source)
    #     self.classname = classname
    #     self.imagesize = imagesize
    #     self.shot = shot
    #     self.iterate = iterate
    #     self.transform = transform
    #     self.target_transform = target_transform
    #
    #     # 调试信息：检查输入参数
    #     print(f"初始化 VisADataset: source={self.source}, classname={self.classname}, imagesize={self.imagesize}")
    #     print(f"Transform: {self.transform}, Target Transform: {self.target_transform}")
    #
    #     # 加载 CSV 数据
    #     csv_path = self.source / "split_csv" / "1cls.csv"
    #     print(f"尝试加载 CSV 文件: {csv_path}")
    #     if not csv_path.exists():
    #         raise FileNotFoundError(f"未找到 CSV 文件: {csv_path}")
    #     self.csv_data = pd.read_csv(csv_path)
    #     print(f"CSV 文件加载成功，样本总数: {len(self.csv_data)}")
    #
    #     # 筛选当前类别的数据
    #     self.class_data = self.csv_data[self.csv_data['object'] == self.classname].reset_index(drop=True)
    #     print(f"类别 '{classname}' 加载的样本数量: {len(self.class_data)}")
    #
    #     # 初始化 few-shot 相关属性
    #     self.fewshot_abnorm_img = []
    #     self.fewshot_abnorm_mask = []
    #     self.fewshot_norm_img = []
    #
    #     # 加载 few-shot 样本
    #     self._load_fewshot_samples()
    def __init__(self, source, classname, imagesize, shot=4, iterate=0):
        self.source = Path(source)
        self.classname = classname
        self.imagesize = imagesize
        self.shot = shot  # few-shot 的样本数量
        self.iterate = iterate
        self.transform = None  # 根据需要可以设置
        self.target_transform = None

        # 加载 CSV 文件
        csv_path = self.source / "split_csv" / "1cls.csv"
        print(f"尝试加载 CSV 文件: {csv_path}")
        self.data_df = pd.read_csv(csv_path)
        print(f"CSV 文件加载成功，样本总数: {len(self.data_df)}")

        # 筛选指定类别的样本
        self.class_data = self.data_df[self.data_df['object'] == self.classname]
        print(f"类别 '{self.classname}' 加载的样本数量: {len(self.class_data)}")

        # 初始化 few-shot 样本列表
        self.fewshot_norm_img = []
        self.fewshot_abnorm_img = []
        self.fewshot_abnorm_mask = []

        # 加载 few-shot 样本
        self._load_fewshot_samples()

    # def _load_fewshot_samples(self):
    #     print(f"开始加载 few-shot 样本，类别: {self.classname}")
    #     for idx, row in self.class_data.iterrows():
    #         image_path = self.source / row['image']
    #         print(f"处理样本 {idx}: image_path={image_path}, label={row['label']}")
    #         if not image_path.exists():
    #             print(f"警告: 找不到图像文件: {image_path}")
    #             continue
    #
    #         # 加载图像
    #         img = Image.open(image_path).convert("RGB")
    #         img = img.resize((self.imagesize, self.imagesize))
    #         print(f"图像加载并调整大小: type={type(img)}, size={img.size}")
    #
    #         if self.transform:
    #             img = self.transform(img)
    #             print(f"应用 transform 后: type={type(img)}, shape={getattr(img, 'shape', '无 shape 属性')}")
    #         else:
    #             img = T.ToTensor()(img)
    #             print(f"默认转换为张量: type={type(img)}, shape={img.shape}")
    #
    #         # 加载掩码
    #         mask_path_str = row['mask']
    #         print(f"原始 mask 值: {mask_path_str}, type: {type(mask_path_str)}")
    #         if isinstance(mask_path_str, str) and mask_path_str.strip():  # 检查是否为非空字符串
    #             mask_path = self.source / mask_path_str
    #             print(f"掩码路径: {mask_path}")
    #             if mask_path.exists():
    #                 mask = Image.open(mask_path).convert("L")
    #                 mask = mask.resize((self.imagesize, self.imagesize))
    #                 if self.target_transform:
    #                     mask = self.target_transform(mask)
    #                 else:
    #                     mask = T.ToTensor()(mask)
    #                 print(f"掩码加载并转换: type={type(mask)}, shape={mask.shape}")
    #             else:
    #                 mask = torch.zeros((1, self.imagesize, self.imagesize))
    #                 print(f"掩码文件不存在，创建零掩码: type={type(mask)}, shape={mask.shape}")
    #         else:
    #             mask = torch.zeros((1, self.imagesize, self.imagesize))
    #             print(f"mask 值无效（非字符串或空），创建零掩码: type={type(mask)}, shape={mask.shape}")
    #
    #         # 添加到列表
    #         if row['label'] == 'normal':
    #             self.fewshot_norm_img.append(img)
    #             abnorm_img = torch.zeros_like(img)
    #             self.fewshot_abnorm_img.append(abnorm_img)
    #             self.fewshot_abnorm_mask.append(mask)
    #             print(
    #                 f"添加正常样本: img shape={img.shape}, abnorm_img shape={abnorm_img.shape}, mask shape={mask.shape}")
    #         else:
    #             self.fewshot_abnorm_img.append(img)
    #             self.fewshot_abnorm_mask.append(mask)
    #             print(f"添加异常样本: img shape={img.shape}, mask shape={mask.shape}")
    #
    #     # 转换为张量
    #     print(f"开始转换 few-shot 数据为张量")
    #     if self.fewshot_norm_img:
    #         self.fewshot_norm_img = torch.stack(self.fewshot_norm_img)
    #         print(f"fewshot_norm_img 转换为张量: shape={self.fewshot_norm_img.shape}")
    #     else:
    #         self.fewshot_norm_img = torch.empty((0, 3, self.imagesize, self.imagesize))
    #         print(f"fewshot_norm_img 为空: shape={self.fewshot_norm_img.shape}")
    #
    #     if self.fewshot_abnorm_img:
    #         self.fewshot_abnorm_img = torch.stack(self.fewshot_abnorm_img)
    #         self.fewshot_abnorm_mask = torch.stack(self.fewshot_abnorm_mask)
    #         print(f"fewshot_abnorm_img 转换为张量: shape={self.fewshot_abnorm_img.shape}")
    #         print(f"fewshot_abnorm_mask 转换为张量: shape={self.fewshot_abnorm_mask.shape}")
    #     else:
    #         self.fewshot_abnorm_img = torch.empty((0, 3, self.imagesize, self.imagesize))
    #         self.fewshot_abnorm_mask = torch.empty((0, 1, self.imagesize, self.imagesize))
    #         print(f"fewshot_abnorm_img 为空: shape={self.fewshot_abnorm_img.shape}")
    #         print(f"fewshot_abnorm_mask 为空: shape={self.fewshot_abnorm_mask.shape}")
    #
    #     print("Normal images shape:", self.fewshot_norm_img.shape if len(self.fewshot_norm_img) else "empty")
    #     print("Abnormal images shape:", self.fewshot_abnorm_img.shape if len(self.fewshot_abnorm_img) else "empty")
    #     print("Abnormal masks shape:", self.fewshot_abnorm_mask.shape if len(self.fewshot_abnorm_mask) else "empty")

    def _load_fewshot_samples(self):
        """加载 few-shot 样本，限制每个类别为 self.shot 个样本"""
        print(f"开始加载 few-shot 样本，类别: {self.classname}, shot: {self.shot}")
        norm_count = 0
        abnorm_count = 0

        # 遍历数据，直到加载足够的样本
        for idx, row in self.class_data.iterrows():
            # 检查是否已加载足够的样本
            if norm_count >= self.shot and abnorm_count >= self.shot:
                break

            image_path = self.source / row['image']
            print(f"处理样本 {idx}: image_path={image_path}, label={row['label']}")
            if not image_path.exists():
                print(f"警告: 找不到图像文件: {image_path}")
                continue

            # 加载图像
            img = Image.open(image_path).convert("RGB")
            img = img.resize((self.imagesize, self.imagesize))
            print(f"图像加载并调整大小: type={type(img)}, size={img.size}")

            if self.transform:
                img = self.transform(img)
            else:
                img = T.ToTensor()(img)
            print(f"默认转换为张量: type={type(img)}, shape={img.shape}")

            # 加载掩码
            mask_path_str = row['mask']
            print(f"原始 mask 值: {mask_path_str}, type: {type(mask_path_str)}")
            if isinstance(mask_path_str, str) and mask_path_str.strip():
                mask_path = self.source / mask_path_str
                print(f"掩码路径: {mask_path}")
                if mask_path.exists():
                    mask = Image.open(mask_path).convert("L")
                    mask = mask.resize((self.imagesize, self.imagesize))
                    if self.target_transform:
                        mask = self.target_transform(mask)
                    else:
                        mask = T.ToTensor()(mask)
                    print(f"掩码加载并转换: type={type(mask)}, shape={mask.shape}")
                else:
                    mask = torch.zeros((1, self.imagesize, self.imagesize))
                    print(f"掩码文件不存在，创建零掩码: type={type(mask)}, shape={mask.shape}")
            else:
                mask = torch.zeros((1, self.imagesize, self.imagesize))
                print(f"mask 值无效（非字符串或空），创建零掩码: type={type(mask)}, shape={mask.shape}")

            # 根据标签添加样本
            if row['label'] == 'normal' and norm_count < self.shot:
                self.fewshot_norm_img.append(img)
                abnorm_img = torch.zeros_like(img)
                self.fewshot_abnorm_img.append(abnorm_img)
                self.fewshot_abnorm_mask.append(mask)
                print(
                    f"添加正常样本: img shape={img.shape}, abnorm_img shape={abnorm_img.shape}, mask shape={mask.shape}")
                norm_count += 1
            elif row['label'] != 'normal' and abnorm_count < self.shot:
                self.fewshot_abnorm_img.append(img)
                self.fewshot_abnorm_mask.append(mask)
                print(f"添加异常样本: img shape={img.shape}, mask shape={mask.shape}")
                abnorm_count += 1

        # 转换为张量
        print(f"开始转换 few-shot 数据为张量")
        if self.fewshot_norm_img:
            self.fewshot_norm_img = torch.stack(self.fewshot_norm_img)
            print(f"fewshot_norm_img 转换为张量: shape={self.fewshot_norm_img.shape}")
        else:
            self.fewshot_norm_img = torch.empty((0, 3, self.imagesize, self.imagesize))
            print(f"fewshot_norm_img 为空: shape={self.fewshot_norm_img.shape}")

        if self.fewshot_abnorm_img:
            self.fewshot_abnorm_img = torch.stack(self.fewshot_abnorm_img)
            self.fewshot_abnorm_mask = torch.stack(self.fewshot_abnorm_mask)
            print(f"fewshot_abnorm_img 转换为张量: shape={self.fewshot_abnorm_img.shape}")
            print(f"fewshot_abnorm_mask 转换为张量: shape={self.fewshot_abnorm_mask.shape}")
        else:
            self.fewshot_abnorm_img = torch.empty((0, 3, self.imagesize, self.imagesize))
            self.fewshot_abnorm_mask = torch.empty((0, 1, self.imagesize, self.imagesize))
            print(f"fewshot_abnorm_img 为空: shape={self.fewshot_abnorm_img.shape}")
            print(f"fewshot_abnorm_mask 为空: shape={self.fewshot_abnorm_mask.shape}")

        print("Normal images shape:", self.fewshot_norm_img.shape if len(self.fewshot_norm_img) else "empty")
        print("Abnormal images shape:", self.fewshot_abnorm_img.shape if len(self.fewshot_abnorm_img) else "empty")
        print("Abnormal masks shape:", self.fewshot_abnorm_mask.shape if len(self.fewshot_abnorm_mask) else "empty")

        # 检查是否加载了足够的样本
        if len(self.fewshot_norm_img) < self.shot or len(self.fewshot_abnorm_img) < self.shot:
            raise ValueError(
                f"未加载足够的 few-shot 样本: normal={len(self.fewshot_norm_img)}, abnormal={len(self.fewshot_abnorm_img)}, 需要 {self.shot}")

    # def __getitem__(self, idx):
    #     classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
    #     print(f"__getitem__ 调用: idx={idx}, image_path={image_path}, mask_path={mask_path}")
    #
    #     # 加载图像
    #     image = PIL.Image.open(image_path).convert("RGB")
    #     original_img_width, original_img_height = image.size
    #     print(f"图像加载: type={type(image)}, size=({original_img_width}, {original_img_height})")
    #
    #     # 应用图像变换
    #     image = self.transform_img[0](image)
    #     print(f"应用 transform_img 后: type={type(image)}, shape={getattr(image, 'shape', '无 shape 属性')}")
    #
    #     # 加载掩码（如果存在）
    #     if mask_path is not None:
    #         mask = PIL.Image.open(mask_path)
    #         mask = (pil_to_tensor(mask) != 0).float()
    #         mask = self.transform_mask[0](mask)
    #         print(f"掩码加载并转换: type={type(mask)}, shape={mask.shape}")
    #     else:
    #         mask = torch.zeros([1, self.imagesize, self.imagesize])
    #         print(f"无掩码文件，创建零掩码: type={type(mask)}, shape={mask.shape}")
    #
    #     return {
    #         "image": image,
    #         "mask": mask,
    #         "classname": classname,
    #         "anomaly": anomaly,
    #         "is_anomaly": int(anomaly != "normal"),
    #         "image_name": Path(image_path).stem,
    #         "image_path": str(image_path),
    #         "mask_path": str(mask_path) if mask_path else "None",
    #         "original_img_height": original_img_height,
    #         "original_img_width": original_img_width,
    #     }

    def __len__(self):
        return len(self.class_data)

    def __getitem__(self, idx):
        row = self.class_data.iloc[idx]
        image_path = self.source / row['image']
        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.imagesize, self.imagesize))
        img = T.ToTensor()(img) if self.transform is None else self.transform(img)

        mask_path_str = row['mask']
        if isinstance(mask_path_str, str) and mask_path_str.strip():
            mask_path = self.source / mask_path_str
            if mask_path.exists():
                mask = Image.open(mask_path).convert("L")
                mask = mask.resize((self.imagesize, self.imagesize))
                mask = T.ToTensor()(mask) if self.target_transform is None else self.target_transform(mask)
            else:
                mask = torch.zeros((1, self.imagesize, self.imagesize))
        else:
            mask = torch.zeros((1, self.imagesize, self.imagesize))

        return {
            'image': img,
            'mask': mask,
            'is_anomaly': 1 if row['label'] != 'normal' else 0
        }

    def get_image_data(self):
        """获取图像数据路径"""
        print(f"调用 get_image_data，加载所有类别的数据路径")
        imgpaths_per_class = defaultdict(lambda: defaultdict(list))
        maskpaths_per_class = defaultdict(lambda: defaultdict(list))

        # 对于每个类别
        for classname in self.classnames_to_use:
            class_path = self.source / classname / "Data"
            print(f"处理类别: {classname}, class_path={class_path}")

            # 加载正常图像
            normal_path = class_path / "Images" / "Normal"
            if normal_path.exists():
                normal_images = list(normal_path.glob("*.JPG"))
                print(f"找到 {len(normal_images)} 个正常图像在 {normal_path}")
                for img_path in normal_images:
                    imgpaths_per_class[classname]["good"].append(img_path)
                    maskpaths_per_class[classname]["good"].append(None)

            # 加载异常图像和掩码
            anomaly_img_path = class_path / "Images" / "Anomaly"
            anomaly_mask_path = class_path / "Masks" / "Anomaly"
            if anomaly_img_path.exists() and anomaly_mask_path.exists():
                anomaly_images = list(anomaly_img_path.glob("*.JPG"))
                print(f"找到 {len(anomaly_images)} 个异常图像在 {anomaly_img_path}")
                for img_path in anomaly_images:
                    mask_path = anomaly_mask_path / img_path.name
                    if mask_path.exists():
                        imgpaths_per_class[classname]["anomaly"].append(img_path)
                        maskpaths_per_class[classname]["anomaly"].append(mask_path)
                    else:
                        print(f"警告: 找不到对应的掩码文件: {mask_path}")

        # 创建数据列表
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)
        print(f"数据路径加载完成，总样本数: {len(data_to_iterate)}")

        return dict(imgpaths_per_class), dict(maskpaths_per_class), data_to_iterate






