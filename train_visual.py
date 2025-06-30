import os
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from dataset.medical_few import MedDataset
from CLIP.clip import create_model
from CLIP.tokenizer import tokenize
# from CLIP.adapter import CLIP_Inplanted
# from CLIP.adaptercbamconv import CLIP_Inplanted
# from CLIP.adapter_deepth_conv import CLIP_Inplanted
from CLIP.mdecoder import CLIP_Inplanted
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise
from loss import FocalLoss, BinaryDiceLoss
# from utils_textmore import augment, cos_sim, encode_text_with_prompt_ensemble
from utils import augment, cos_sim, encode_text_with_prompt_ensemble
# from utilscopy import augment, cos_sim, encode_text_with_prompt_ensemble
from prompt import REAL_NAME
from plots import plot_segmentation_images
import cv2
import matplotlib.pyplot as plt
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain': 3, 'Liver': 2, 'Retina_RESC': 1, 'Retina_OCT2017': -1, 'Chest': -2, 'Histopathology': -3}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")
    parser.add_argument('--obj', type=str, default='Retina_RESC')  # default='Retina_RESC'
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/vision/')  # default='./ckpt/few-shot/few-shot/'
    #parser.add_argument('--save_paths', type=str, default='./ckpt/liver/')
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument("--epoch", type=int, default=50, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    args = parser.parse_args()

    setup_seed(args.seed)

    # fixed feature extractor
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device,
                              pretrained=args.pretrain, require_pretrained=True)
    clip_model.eval()

    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
    model.eval()

    checkpoint = torch.load(os.path.join(f'{args.save_path}', f'{args.obj}.pth'))
    model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
    model.det_adapters.load_state_dict(checkpoint["det_adapters"])

    for name, param in model.named_parameters():
        param.requires_grad = True

    # optimizer for only adapters
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # load test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # few-shot image augmentation
    augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)

    augment_fewshot_label = torch.cat(
        [torch.Tensor([1] * len(augment_abnorm_img)), torch.Tensor([0] * len(augment_normal_img))], dim=0)

    train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

    # memory bank construction
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[args.obj], device)

    best_result = 0

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
    seg_mem_features = [torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0) for i in
                        range(len(seg_features[0]))]
    det_mem_features = [torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0) for i in
                        range(len(det_features[0]))]

    result = test(args, model, test_loader, text_features, seg_mem_features, det_mem_features,save_path='./ckpt/res/')


def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features,
         visualize=True, save_path=None):
    gt_list = []
    gt_mask_list = []
    det_image_scores_zero = []
    det_image_scores_few = []
    seg_score_map_zero = []
    seg_score_map_few = []
    images_list = []
    pred_masks_list = []

    for idx, (image, y, mask) in enumerate(tqdm(test_loader)):
        image = image.to(device).to(torch.float16)  # 假设使用半精度
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, seg_patch_tokens, det_patch_tokens = model(image)
            seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
            det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

            if CLASS_INDEX[args.obj] > 0:
                # few-shot, seg head
                anomaly_maps_few_shot = []
                for idx, p in enumerate(seg_patch_tokens):
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
                    seg_patch_tokens[layer] /= seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
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
                    det_patch_tokens[layer] /= det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features).unsqueeze(0)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score += anomaly_map.mean()
                det_image_scores_zero.append(anomaly_score.cpu().numpy())

            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())

            if visualize:
                images_list.append(image[0].cpu().numpy())
                if CLASS_INDEX[args.obj] > 0:
                    pred_masks_list.append(0.5 * score_map_zero + 0.5 * score_map_few)
                else:
                    pred_masks_list.append(anomaly_map_few_shot)

    gt_list = np.array(gt_list)
    gt_mask_list = np.asarray(gt_mask_list)
    gt_mask_list = (gt_mask_list > 0).astype(np.int_)

    if CLASS_INDEX[args.obj] > 0:
        seg_score_map_zero = np.array(seg_score_map_zero)
        seg_score_map_few = np.array(seg_score_map_few)

        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / \
                             (seg_score_map_zero.max() - seg_score_map_zero.min())
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / \
                            (seg_score_map_few.max() - seg_score_map_few.min())

        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'{args.obj} pAUC : {round(seg_roc_auc, 4)}')

        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'{args.obj} AUC : {round(roc_auc_im, 4)}')

        if visualize:
            visualize_results(images_list, gt_mask_list, pred_masks_list,
                              is_segmentation=True, save_path=save_path)

        return seg_roc_auc + roc_auc_im
    else:
        det_image_scores_zero = np.array(det_image_scores_zero)
        det_image_scores_few = np.array(det_image_scores_few)

        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / \
                                (det_image_scores_zero.max() - det_image_scores_zero.min())
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / \
                               (det_image_scores_few.max() - det_image_scores_few.min())

        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f'{args.obj} AUC : {round(img_roc_auc_det, 4)}')

        if visualize:
            visualize_results(images_list, gt_mask_list, pred_masks_list,
                              is_segmentation=False, save_path=save_path)

        return img_roc_auc_det


def enhance_contrast(mask, gamma=0.5):
    """增强掩码对比度"""
    mask = np.clip(mask, 0, 1)
    return mask ** gamma



def visualize_results(images, gt_masks, pred_masks, is_segmentation, save_path=None,
                      max_samples=1493,
                      image_transform=lambda x: x, mask_transform=lambda x: x):
    """
    可视化检测和分割结果，支持保存不同类型图片到不同文件夹。
    """

    if isinstance(save_path, str):
        save_path = Path(save_path)
    if save_path:
        subdirs = ['input_images', 'pred_masks', 'anomaly_maps', 'overlays', 'results_combined']
        if is_segmentation:
            subdirs.append('gt_masks')
        for subdir in subdirs:
            (save_path / subdir).mkdir(parents=True, exist_ok=True)

    num_samples = min(len(images), max_samples)
    print(f"Visualizing and saving {num_samples} samples...")

    for i in range(num_samples):
        img = images[i]
        if img.ndim == 3 and img.shape[0] in [1, 3]:  # (C,H,W)
            img = np.transpose(img, (1, 2, 0))
        img = image_transform(img)
        img = (img - img.min()) / (img.max() - img.min())
        img = img.astype(np.float32)

        gt_mask = None
        if is_segmentation and gt_masks is not None and len(gt_masks) > i:
            gt_mask = mask_transform(gt_masks[i])
            # gt_mask = (gt_mask - gt_mask.min()) / (gt_mask.max() - gt_mask.min())
            # 只对非0/1掩码做归一化
            eps = 1e-8
            gt_min, gt_max = gt_mask.min(), gt_mask.max()
            if gt_max - gt_min > eps:
                gt_mask = (gt_mask - gt_min) / (gt_max - gt_min)
            else:
                gt_mask = gt_mask.astype(np.float32)

        pred_mask = pred_masks[i]
        if isinstance(pred_mask, np.ndarray):
            pred_mask = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min())
            if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
                pred_mask = pred_mask[0]
        else:
            pred_mask = np.array(pred_mask)
            pred_mask = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min())
            if pred_mask.ndim == 3 and pred_mask.shape[0] == 1:
                pred_mask = pred_mask[0]
        pred_mask = pred_mask.astype(np.float32)

        # 转为 uint8 并生成 heatmap 和 overlay
        pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(pred_mask_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(heatmap, 0.5, (img * 255).astype(np.uint8), 0.5, 0)

        binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        if is_segmentation:
            fig, axes = plt.subplots(1, 5, figsize=(18, 4))
            axes[0].imshow(img)
            axes[0].set_title("Input Image")

            axes[1].imshow(gt_mask, cmap='gray')
            axes[1].set_title("Ground Truth Mask")

            axes[2].imshow(pred_mask, cmap='jet')
            axes[2].set_title("Anomaly Map")

            axes[3].imshow(overlay)
            axes[3].set_title("Overlay")

            axes[4].imshow(binary_mask, cmap='gray')
            axes[4].set_title("Predicted Mask (Raw)")

            for ax in axes:
                ax.axis('off')
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img)
            axes[0].set_title("Input Image")
            axes[1].imshow(pred_mask, cmap='jet')
            axes[1].set_title("Anomaly Map")
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay")
            for ax in axes:
                ax.axis('off')

        plt.tight_layout()

        # 保存图像
        if save_path:
            # 保存输入图像
            cv2.imwrite(str(save_path / 'input_images' / f'img_{i}.png'), (img * 255).astype(np.uint8))

            binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
            cv2.imwrite(str(save_path / 'pred_masks' / f'pred_{i}.png'), binary_mask)
            # 保存预测掩码（灰度图）
            #cv2.imwrite(str(save_path / 'pred_masks' / f'pred_{i}.png'), (pred_mask * 255).astype(np.uint8))

            # 保存伪彩色热力图
            cv2.imwrite(str(save_path / 'anomaly_maps' / f'heatmap_{i}.png'), heatmap)

            # 保存叠加图
            cv2.imwrite(str(save_path / 'overlays' / f'overlay_{i}.png'), overlay)
            print(gt_mask)
            print(is_segmentation)
            # 保存GT掩码（如有）
            # if is_segmentation and gt_mask is not None:
            #     gt_mask_vis = (gt_mask * 255).astype(np.uint8) if gt_mask.max() <= 1 else gt_mask.astype(np.uint8)
            #     cv2.imwrite(str(save_path / 'gt_masks' / f'gt_{i}.png'), gt_mask_vis)
            # 保存GT掩码（如有）
            if is_segmentation and gt_mask is not None:
                gt_mask_vis = (gt_mask * 255).astype(np.uint8)  # 0~1 转为 0~255
                cv2.imwrite(str(save_path / 'gt_masks' / f'gt_{i}.png'), gt_mask_vis)

            # 保存合并结果图（matplotlib形式）
            fig.savefig(str(save_path / 'results_combined' / f'result_{i}.png'), bbox_inches='tight')

        plt.close()

if __name__ == '__main__':
    main()