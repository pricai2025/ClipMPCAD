import os
import argparse
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
# 修改：导入适用于 MVTec 数据集的加载器
from datasets.mvtec import MVTecDataset  # 需要你自己实现
#from datasets.mpdd import MVTecDataset
from CLIP.clip import create_model
from CLIP.tokenizer import tokenize
#from CLIP.adaptercbamconv import CLIP_Inplanted
from CLIP.mdecoder import CLIP_Inplanted
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise
from loss import FocalLoss, BinaryDiceLoss
#from utilscopy import augment, cos_sim, encode_text_with_prompt_ensemble
from utils import augment, cos_sim, encode_text_with_prompt_ensemble
from prompt import REAL_NAME
from pathlib import Path

# 禁用分词器的并行处理警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings("ignore")

# 设置设备（GPU/CPU）
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:3" if use_cuda else "cpu")

# # 更新：添加 MVTec 类别映射
CLASS_INDEX = {'bottle': 0, 'cable': 1, 'capsule': 2, 'carpet': 3, 'grid': 4, 'hazelnut': 5, 'leather': 6,
               'metal_nut': 7,
               'pill': 8, 'screw': 9, 'tile': 10, 'toothbrush': 11, 'transistor': 12, 'wood': 13,
               'zipper': 14}  # 示例，需根据实际类别调整
# 修改类别索引为 MPDD 的类别
# CLASS_INDEX = {
#     'bracket_black': 0,
#     'bracket_brown': 1,
#     'bracket_white': 2,
#     'connector': 3,
#     'metal_plate': 4,
#     'tubes': 5
# }


def setup_seed(seed):
    """设置随机种子，确保实验可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_single_class(args):
    """训练单个类别的函数"""
    # 数据路径处理和验证
    args.data_path = str(Path(args.data_path).resolve())
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"找不到数据路径: {args.data_path}")
    
    class_path = Path(args.data_path) / args.obj
    if not class_path.exists():
        raise FileNotFoundError(f"找不到类别路径: {class_path}")
    
    print(f"正在加载数据集: {args.data_path}")
    print(f"类别: {args.obj}")
    print(f"图像大小: {args.img_size}")
    
    # 设置随机种子
    setup_seed(args.seed)

    # 初始化 CLIP 模型作为特征提取器
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, 
                            device=device, pretrained=args.pretrain, require_pretrained=True)
    clip_model.eval()  # 设置为评估模式

    # 初始化带适配器的模型
    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
    model.eval()

    # 设置所有参数可训练
    for name, param in model.named_parameters():
        param.requires_grad = True

    # 为分割和检测适配器分别创建优化器
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), 
                                   lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), 
                                   lr=args.learning_rate, betas=(0.5, 0.999))

    # 加载数据集和创建数据加载器
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MVTecDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                                            shuffle=False, **kwargs)

    # 对少样本数据进行数据增强
    augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, 
                                                     test_dataset.fewshot_abnorm_mask)
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

    # 合并正常和异常样本
    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)

    # 创建标签：异常样本标记为1，正常样本标记为0
    augment_fewshot_label = torch.cat([
        torch.Tensor([1] * len(augment_abnorm_img)), 
        torch.Tensor([0] * len(augment_normal_img))
    ], dim=0)

    # 创建训练数据集和加载器
    train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, 
                                                 augment_fewshot_mask, 
                                                 augment_fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, 
                                             shuffle=True, **kwargs)

    # 构建记忆库的数据加载器（仅使用正常样本）
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, 
                                               shuffle=True, **kwargs)

    # 定义损失函数
    loss_focal = FocalLoss()        # 用于分割任务的焦点损失
    loss_dice = BinaryDiceLoss()    # 用于分割任务的Dice损失
    loss_bce = torch.nn.BCEWithLogitsLoss()  # 用于检测任务的二元交叉熵损失

    # 编码文本提示
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[args.obj], device)
        text_features = text_features.float()  # 确保数据类型一致

    # 只训练一轮，不需要记录最佳结果
    for epoch in range(args.epoch):
        print('epoch ', epoch, ':')
        loss_list = []

        # 训练循环
        for (image, gt, label) in train_loader:
            image = image.to(device)
            
            with torch.cuda.amp.autocast():
                # 前向传播，获取特征
                _, seg_patch_tokens, det_patch_tokens = model(image)
                # 提取patch tokens（去除CLS token）
                seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
                det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

                # 计算检测损失
                det_loss = 0
                image_label = label.to(device)
                for layer in range(len(det_patch_tokens)):
                    # 特征归一化
                    det_patch_tokens[layer] = det_patch_tokens[layer] / det_patch_tokens[layer].norm(
                        dim=-1, keepdim=True)
                    # 计算异常图
                    anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features).unsqueeze(0)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    # 累加检测损失
                    det_loss += loss_bce(anomaly_score, image_label)

                # 如果当前类别支持分割
                if CLASS_INDEX.get(args.obj) is not None:
                    # 计算像素级别的分割损失
                    seg_loss = 0
                    mask = gt.squeeze(0).to(device)
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0  # 二值化掩码
                    
                    for layer in range(len(seg_patch_tokens)):
                        # 特征归一化
                        seg_patch_tokens[layer] = seg_patch_tokens[layer] / seg_patch_tokens[layer].norm(
                            dim=-1, keepdim=True)
                        # 计算异常图
                        anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_features).unsqueeze(0)
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        # 调整大小并重塑为图像格式
                        anomaly_map = F.interpolate(
                            anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                            size=args.img_size, mode='bilinear', align_corners=True)
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        # 计算分割损失（焦点损失和Dice损失）
                        seg_loss += loss_focal(anomaly_map, mask)
                        seg_loss += loss_dice(anomaly_map[:, 1, :, :], mask)

                    # 总损失为分割损失和检测损失之和
                    loss = seg_loss + det_loss
                    loss.requires_grad_(True)
                    # 优化器步骤
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    loss.backward()
                    seg_optimizer.step()
                    det_optimizer.step()
                else:
                    # 如果不支持分割，只使用检测损失
                    loss = det_loss
                    loss.requires_grad_(True)
                    det_optimizer.zero_grad()
                    loss.backward()
                    det_optimizer.step()

                loss_list.append(loss.item())

        # 打印平均损失
        print("Loss: ", np.mean(loss_list))

        # 构建特征记忆库
        seg_features = []
        det_features = []
        for image in support_loader:
            image = image[0].to(device)
            with torch.no_grad():
                _, seg_patch_tokens, det_patch_tokens = model(image)
                seg_patch_tokens = [p[0].contiguous() for p in seg_patch_tokens]
                det_patch_tokens = [p[0].contiguous() for p in det_patch_tokens]
                seg_features.append(seg_patch_tokens)
                det_features.append(det_patch_tokens)
        
        # 合并所有特征
        seg_mem_features = [torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0) 
                          for i in range(len(seg_features[0]))]
        det_mem_features = [torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0) 
                          for i in range(len(det_features[0]))]

        # 评估当前模型并直接返回结果
        result = test(args, model, test_loader, text_features, seg_mem_features, det_mem_features)
        
        # 保存模型（如果需要）
        if args.save_model == 1:
            ckp_path = os.path.join(args.save_path, f'{args.obj}.pth')
            torch.save({
                'seg_adapters': model.seg_adapters.state_dict(),
                'det_adapters': model.det_adapters.state_dict()
            }, ckp_path)

        # 直接返回最后一轮的结果
        return result


def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    """测试函数：根据类别使用不同的评估标准"""
    gt_list = []
    gt_mask_list = []  # 用于存储掩码真值
    det_image_scores_zero = []
    det_image_scores_few = []
    seg_score_map_zero = []
    seg_score_map_few = []

    for data in tqdm(test_loader):
        image = data['image'].to(device)
        mask = data['mask']
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features, seg_patch_tokens, det_patch_tokens = model(image)
            seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
            det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

            # 根据类别选择不同的评估方式
            if CLASS_INDEX.get(args.obj) is not None:
                # 对支持分割的类别进行分割评估
                # few-shot, seg head
                anomaly_maps_few_shot = []
                for idx, p in enumerate(seg_patch_tokens):
                    p = p / p.norm(dim=-1, keepdim=True)  # 归一化
                    cos = cos_sim(seg_mem_features[idx], p)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                    anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                       size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
                seg_score_map_few.append(score_map_few)

                # zero-shot, seg head
                anomaly_maps = []
                for layer in range(len(seg_patch_tokens)):
                    seg_patch_tokens[layer] = seg_patch_tokens[layer] / seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_features).unsqueeze(0)
                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                              size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                    anomaly_maps.append(anomaly_map.cpu().numpy())
                score_map_zero = np.sum(anomaly_maps, axis=0)
                seg_score_map_zero.append(score_map_zero)

            else:
                # 对不支持分割的类别进行检测评估
                # few-shot, det head
                anomaly_maps_few_shot = []
                for idx, p in enumerate(det_patch_tokens):
                    cos = cos_sim(det_mem_features[idx], p)
                    height = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                    anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                       size=args.img_size, mode='bilinear', align_corners=True)
                    anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
                anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
                score_few_det = anomaly_map_few_shot.mean()
                det_image_scores_few.append(score_few_det)

                # zero-shot, det head
                anomaly_score = 0
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] = det_patch_tokens[layer] / det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features).unsqueeze(0)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score += anomaly_map.mean()
                det_image_scores_zero.append(anomaly_score.cpu().numpy())

            gt_mask_list.append(mask.squeeze().cpu().numpy())
            gt_list.append(data['is_anomaly'].cpu().numpy())

    # 转换为 numpy 数组并计算评估指标
    gt_list = np.array(gt_list)
    gt_mask_list = np.array(gt_mask_list)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)

    if CLASS_INDEX.get(args.obj) is not None:
        # 对支持分割的类别计算分割和图像级别的 AUC
        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)

        # 归一化分数
        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (
                seg_score_map_zero.max() - seg_score_map_zero.min())
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (
                seg_score_map_few.max() - seg_score_map_few.min())

        # 组合分数
        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        
        # 计算像素级 AUC
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc, 4)}')

        # 计算图像级 AUC
        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'{args.obj} AUC : {round(roc_auc_im, 4)}')

        return seg_roc_auc, roc_auc_im  # 返回两个值

    else:
        # 对不支持分割的类别只计算检测 AUC
        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)

        # 归一化分数
        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (
                det_image_scores_zero.max() - det_image_scores_zero.min())
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (
                det_image_scores_few.max() - det_image_scores_few.min())

        # 组合分数
        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        
        # 计算检测 AUC
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f'{args.obj} AUC : {round(img_roc_auc_det, 4)}')

        return img_roc_auc_det  # 只返回一个值


def main():
    """主函数：处理参数并遍历所有类别"""
    parser = argparse.ArgumentParser(description='Testing')
    # 添加各种命令行参数
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="选择模型架构")
    parser.add_argument('--pretrain', type=str, default='openai', help="预训练模型来源")
    parser.add_argument('--obj', type=str, default='bottle', help="目标类别")
    parser.add_argument('--data_path', type=str, default='./data/mvtec/', help="数据集路径")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/1/')
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="学习率")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="使用的特征层")
    parser.add_argument('--seed', type=int, default=111) # 111
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--iterate', type=int, default=0)
    args = parser.parse_args()
    
    # 修改为只训练一轮
    args.epoch = 1
    
    # 创建保存结果的字典
    results = {
        'image_auc': {},  # 存储图像级别的 AUC
        'pixel_auc': {}   # 存储像素级别的 AUC（pAUC）
    }
    
    # 遍历所有类别
    for class_name in CLASS_INDEX.keys():
        print(f"\n{'='*50}")
        print(f"开始训练类别: {class_name}")
        print(f"{'='*50}")
        
        # 更新当前类别
        args.obj = class_name
        
        # 创建类别特定的保存路径
        class_save_path = os.path.join(args.save_path, class_name)
        os.makedirs(class_save_path, exist_ok=True)
        args.save_path = class_save_path
        
        try:
            # 训练当前类别
            result = train_single_class(args)
            if isinstance(result, tuple):
                # 如果返回的是元组（支持分割的类别）
                pixel_auc, image_auc = result
                results['pixel_auc'][class_name] = pixel_auc
                results['image_auc'][class_name] = image_auc
                print(f"\n类别 {class_name}:")
                print(f"Image AUC: {image_auc:.4f}")
                print(f"Pixel AUC: {pixel_auc:.4f}")
            else:
                # 如果只返回一个值（不支持分割的类别）
                results['image_auc'][class_name] = result
                print(f"\n类别 {class_name}:")
                print(f"Image AUC: {result:.4f}")
        except Exception as e:
            print(f"\n类别 {class_name} 训练失败: {str(e)}")
            results['image_auc'][class_name] = None
            results['pixel_auc'][class_name] = None
    
    # 打印所有结果和计算平均值
    print("\n" + "="*50)
    print("所有类别训练完成")
    print("="*50)
    
    # 计算图像级别 AUC 的平均值
    valid_image_aucs = [v for v in results['image_auc'].values() if v is not None]
    if valid_image_aucs:
        avg_image_auc = sum(valid_image_aucs) / len(valid_image_aucs)
        print(f"\n图像级别 AUC 结果:")
        for class_name, auc in results['image_auc'].items():
            if auc is not None:
                print(f"{class_name}: {auc:.4f}")
        print(f"平均 Image AUC: {avg_image_auc:.4f}")
    
    # 计算像素级别 AUC 的平均值
    valid_pixel_aucs = [v for v in results['pixel_auc'].values() if v is not None]
    if valid_pixel_aucs:
        avg_pixel_auc = sum(valid_pixel_aucs) / len(valid_pixel_aucs)
        print(f"\n像素级别 AUC 结果:")
        for class_name, auc in results['pixel_auc'].items():
            if auc is not None:
                print(f"{class_name}: {auc:.4f}")
        print(f"平均 Pixel AUC: {avg_pixel_auc:.4f}")


if __name__ == "__main__":
    main()
