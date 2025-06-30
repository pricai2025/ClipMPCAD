from collections import defaultdict
from pathlib import Path
import random
import torch
import PIL
from torchvision.transforms.v2.functional import pil_to_tensor
from .base import BaseDataset, DatasetSplit

class BTADDataset(BaseDataset):
    CLASSNAMES = ['01', '02', '03']

    def __init__(self, source: str, classname: str, imagesize: int, shot: int = 1, iterate: int = 0):
        source = Path(source)
        super().__init__(
            source=source,
            classname=classname,
            resize=imagesize,
            split=DatasetSplit.TEST,
            train_val_split=1.0
        )
        self.shot = shot
        self.iterate = iterate
        
        self.fewshot_abnorm_img = []
        self.fewshot_abnorm_mask = []
        self.fewshot_norm_img = []
        
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
                # 同时支持 PNG 和 BMP 格式
                normal_files = sorted(list(normal_path.glob("*.png")) + list(normal_path.glob("*.bmp")))
                if normal_files:
                    selected_normal = random.sample(normal_files, min(self.shot, len(normal_files)))
                    for img_path in selected_normal:
                        img = PIL.Image.open(img_path).convert("RGB")
                        img = self.transform_img[0](img)
                        normal_images.append(img)

            # 加载异常样本
            test_path = self.source / classname / "test"
            if test_path.exists():
                for defect_path in test_path.glob("*"):
                    if defect_path.is_dir() and defect_path.name != "good":
                        # 同时支持 PNG 和 BMP 格式
                        img_files = sorted(list(defect_path.glob("*.png")) + list(defect_path.glob("*.bmp")))
                        mask_path = self.source / classname / "ground_truth" / defect_path.name
                        
                        if img_files and mask_path.exists():
                            mask_files = sorted(list(mask_path.glob("*.png")) + list(mask_path.glob("*.bmp")))
                            paired_files = list(zip(img_files, mask_files))
                            if paired_files:
                                selected_pairs = random.sample(paired_files, min(self.shot, len(paired_files)))
                                for img_path, mask_path in selected_pairs:
                                    img = PIL.Image.open(img_path).convert("RGB")
                                    img = self.transform_img[0](img)
                                    abnormal_images.append(img)
                                    
                                    mask = PIL.Image.open(mask_path).convert("L")  # 转换为灰度图
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
                test_path = self.source / classname / "test"
                if not test_path.exists():
                    print(f"Warning: Test path not found: {test_path}")
                    continue

                for defect_path in test_path.glob("*"):
                    if not defect_path.is_dir():
                        continue
                        
                    defect_type = defect_path.name
                    
                    # 同时支持 PNG 和 BMP 格式
                    image_files = sorted(list(defect_path.glob("*.png")) + list(defect_path.glob("*.bmp")))
                    if not image_files:
                        print(f"Warning: No images found in {defect_path}")
                        continue
                        
                    imgpaths_per_class[classname][defect_type].extend(image_files)
                    
                    if defect_type != "good":
                        mask_path = self.source / classname / "ground_truth" / defect_type
                        if mask_path.exists():
                            mask_files = sorted(list(mask_path.glob("*.png")) + list(mask_path.glob("*.bmp")))
                            if len(mask_files) != len(image_files):
                                print(f"Warning: Mismatch in file counts for {classname}/{defect_type}")
                                min_count = min(len(image_files), len(mask_files))
                                image_files = image_files[:min_count]
                                mask_files = mask_files[:min_count]
                            maskpaths_per_class[classname][defect_type].extend(mask_files)
                        else:
                            print(f"Warning: No mask path found for {classname}/{defect_type}")
                            maskpaths_per_class[classname][defect_type].extend([None] * len(image_files))
                    else:
                        maskpaths_per_class[classname][defect_type].extend([None] * len(image_files))
                    
                    for img_path, mask_path in zip(image_files, 
                                                 maskpaths_per_class[classname][defect_type][-len(image_files):]):
                        data_to_iterate.append([classname, defect_type, img_path, mask_path])

            except Exception as e:
                print(f"Error processing {classname}: {str(e)}")
                raise

        if not data_to_iterate:
            raise RuntimeError(f"No valid data found in {self.source}")

        return dict(imgpaths_per_class), dict(maskpaths_per_class), data_to_iterate 