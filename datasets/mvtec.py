from collections import defaultdict
from pathlib import Path
import random
import torch
import PIL
from torchvision.transforms.v2.functional import pil_to_tensor
from .base import BaseDataset, DatasetSplit

class MVTecDataset(BaseDataset):
    CLASSNAMES = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

    def __init__(self, source: str, classname: str, imagesize: int, shot: int = 4, iterate: int = 0):
        # 确保 source 是 Path 对象
        source = Path(source)
        super().__init__(
            source=source,
            classname=classname,
            resize=imagesize,
            split=DatasetSplit.TEST,  # 固定为测试集
            train_val_split=1.0
        )
        self.shot = shot
        self.iterate = iterate
        
        # 初始化 few-shot 相关属性
        self.fewshot_abnorm_img = []
        self.fewshot_abnorm_mask = []
        self.fewshot_norm_img = []
        
        # 加载 few-shot 样本
        self._load_fewshot_samples()

    def _load_fewshot_samples(self):
        """加载 few-shot 样本"""
        normal_images = []
        abnormal_images = []
        abnormal_masks = []

        for classname in self.classnames_to_use:
            # 加载正常样本
            normal_path = self.source / classname / "train" / "good"
            if normal_path.exists():
                normal_files = sorted(normal_path.glob("*.png"))
                if normal_files:
                    # 随机选择正常样本
                    selected_normal = random.sample(normal_files, min(self.shot, len(normal_files)))
                    for img_path in selected_normal:
                        img = PIL.Image.open(img_path).convert("RGB")
                        img = self.transform_img[0](img)  # 使用第一个转换
                        normal_images.append(img)

            # 加载异常样本
            test_path = self.source / classname / "test"
            if test_path.exists():
                for defect_path in test_path.glob("*"):
                    if defect_path.is_dir() and defect_path.name != "good":
                        img_files = sorted(defect_path.glob("*.png"))
                        mask_path = self.source / classname / "ground_truth" / defect_path.name
                        
                        if img_files and mask_path.exists():
                            mask_files = sorted(mask_path.glob("*.png"))
                            # 确保图像和掩码文件数量匹配
                            paired_files = list(zip(img_files, mask_files))
                            if paired_files:
                                # 随机选择异常样本
                                selected_pairs = random.sample(paired_files, min(self.shot, len(paired_files)))
                                for img_path, mask_path in selected_pairs:
                                    # 加载并转换图像
                                    img = PIL.Image.open(img_path).convert("RGB")
                                    img = self.transform_img[0](img)
                                    abnormal_images.append(img)
                                    
                                    # 加载并转换掩码
                                    mask = PIL.Image.open(mask_path)
                                    mask = pil_to_tensor(mask)
                                    mask = (mask != 0).float()
                                    mask = self.transform_mask[0](mask)
                                    abnormal_masks.append(mask)

        # 转换为张量
        if normal_images:
            self.fewshot_norm_img = torch.stack(normal_images)
        if abnormal_images:
            self.fewshot_abnorm_img = torch.stack(abnormal_images)
        if abnormal_masks:
            self.fewshot_abnorm_mask = torch.stack(abnormal_masks)

        if not (len(self.fewshot_norm_img) and len(self.fewshot_abnorm_img)):
            raise RuntimeError(f"Could not load enough few-shot samples from {self.source}")

    def get_image_data(self):
        """获取图像数据路径"""
        imgpaths_per_class = defaultdict(lambda: defaultdict(list))
        maskpaths_per_class = defaultdict(lambda: defaultdict(list))
        data_to_iterate = []

        for classname in self.classnames_to_use:
            try:
                # 获取测试图像路径
                test_path = self.source / classname / "test"
                if not test_path.exists():
                    print(f"Warning: Test path not found: {test_path}")
                    continue

                # 处理每种缺陷类型
                for defect_path in test_path.glob("*"):
                    if not defect_path.is_dir():
                        continue
                        
                    defect_type = defect_path.name
                    
                    # 获取图像文件
                    image_files = sorted(defect_path.glob("*.png"))
                    if not image_files:
                        print(f"Warning: No images found in {defect_path}")
                        continue
                        
                    imgpaths_per_class[classname][defect_type].extend(image_files)
                    
                    # 获取对应的掩码文件（如果存在）
                    if defect_type != "good":
                        mask_path = self.source / classname / "ground_truth" / defect_type
                        if mask_path.exists():
                            mask_files = sorted(mask_path.glob("*.png"))
                            if len(mask_files) != len(image_files):
                                print(f"Warning: Mismatch in file counts for {classname}/{defect_type}")
                                print(f"Images: {len(image_files)}, Masks: {len(mask_files)}")
                                # 使用较小的数量
                                min_count = min(len(image_files), len(mask_files))
                                image_files = image_files[:min_count]
                                mask_files = mask_files[:min_count]
                            maskpaths_per_class[classname][defect_type].extend(mask_files)
                        else:
                            print(f"Warning: No mask path found for {classname}/{defect_type}")
                            maskpaths_per_class[classname][defect_type].extend([None] * len(image_files))
                    else:
                        # 对于正常样本，没有掩码
                        maskpaths_per_class[classname][defect_type].extend([None] * len(image_files))
                    
                    # 添加到迭代列表
                    for img_path, mask_path in zip(image_files, 
                                                 maskpaths_per_class[classname][defect_type][-len(image_files):]):
                        data_to_iterate.append([classname, defect_type, img_path, mask_path])

            except Exception as e:
                print(f"Error processing {classname}: {str(e)}")
                raise

        if not data_to_iterate:
            raise RuntimeError(f"No valid data found in {self.source}")

        return dict(imgpaths_per_class), dict(maskpaths_per_class), data_to_iterate

    def __len__(self):
        return len(self.data_to_iterate)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        
        # 加载图像
        image = PIL.Image.open(image_path).convert("RGB")
        original_img_width, original_img_height = image.size
        
        # 应用数据增强
        image = self.random_transform_img(image)
        
        # 应用图像变换
        if not isinstance(self.resize, list):
            resize_list = [self.resize]
        else:
            resize_list = self.resize
        
        transformed_images = {}
        transformed_masks = {}
        
        for sz, transform_img in zip(resize_list, self.transform_img):
            transformed_images[sz] = transform_img(image)
        
        # 加载掩码（如果存在）
        if mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = (pil_to_tensor(mask) != 0).float()
        else:
            mask = torch.zeros([1, original_img_height, original_img_width])
        
        # 应用掩码变换
        for sz, transform_mask in zip(resize_list, self.transform_mask):
            transformed_masks[sz] = (transform_mask(mask) > 0.5).float()
        
        # 如果只有一个尺寸，返回单个张量而不是字典
        if not isinstance(self.resize, list):
            transformed_images = next(iter(transformed_images.values()))
            transformed_masks = next(iter(transformed_masks.values()))
        
        return {
            "image": transformed_images,
            "mask": transformed_masks,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": Path(image_path).relative_to(self.source).stem,
            "image_path": str(image_path),
            "mask_path": "None" if mask_path is None else str(mask_path),
            "original_img_height": original_img_height,
            "original_img_width": original_img_width,
        }
