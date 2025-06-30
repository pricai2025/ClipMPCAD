import numpy as np
import torch
import torch.nn.functional as F
import kornia as K
from PIL import Image
from CLIP.tokenizer import tokenize
import json
from pathlib import Path

PROMPT_FILES = {
    'chatgpt3.5': [
        #'prompts/chatgpt3.5_simple_image_prompts.json',
        # 'prompts/chatgpt3.5_medical_prompts1.json',
        # 'prompts/chatgpt3.5_medical_prompts2.json',
        #'prompts/chatgpt3.5_prompt1.json',
        #'prompts/chatgpt3.5_prompt2.json',
        #'prompts/chatgpt3.5_prompt3.json',
        #'prompts/chatgpt3.5_prompt4.json',
        #'prompts/chatgpt3.5_prompt5.json'
    ],
    'winclip': [
        #'prompts/winclip_prompt.json'
    ],
    'manual': [
        #'prompts/manual_prompt_with_classname.json',
        #'prompts/manual_prompt_with_classname2.json',
        # 'prompts/manual_prompt_without_classname.json'
    ],
    'fixed_medical': [
        #'prompts/fixed_medical_prompts.json',
        #'prompts/flexible_medical_prompts.json'
    ]
}
def load_prompts(prompt_path):
    """加载prompt配置文件"""
    try:
        with open(prompt_path, 'r') as f:
            prompts = json.load(f)
            print(f"成功加载提示文件: {prompt_path}, 提示数量: {len(prompts.get('normal', {}).get('prompts', [])) + len(prompts.get('abnormal', {}).get('prompts', []))}")
            return prompts
    except Exception as e:
        print(f"Error loading prompt file {prompt_path}: {e}")
        return {"normal": {"prompts": []}, "abnormal": {"prompts": []}}

def encode_text_with_prompt_ensemble(model, obj, device, use_json_prompts=True, verbose=True):
    """增强版的文本编码函数，整合原有模板和JSON prompts"""
    # 原有的基础prompt
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', 
                    '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', 
                      '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]
    
    # 原有的模板
    original_templates = [
        'a bad photo of a {}.','a low resolution photo of the {}.', 
        'a bad photo of the {}.', 'a cropped photo of the {}.',
        'a bright photo of a {}.', 'a dark photo of the {}.', 
        'a photo of my {}.', 'a photo of the cool {}.',
        'a close-up photo of a {}.', 'a black and white photo of the {}.', 
        'a bright photo of the {}.', 'a cropped photo of a {}.',
        'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 
        'a photo of the {}.', 'a good photo of the {}.',
        'a photo of one {}.', 'a close-up photo of the {}.', 
        'a photo of a {}.', 'a low resolution photo of a {}.',
        'a photo of a large {}.', 'a blurry photo of a {}.', 
        'a jpeg corrupted photo of the {}.', 'a good photo of a {}.',
        'a photo of the small {}.', 'a photo of the large {}.', 
        'a black and white photo of a {}.', 'a dark photo of a {}.',
        'a photo of a cool {}.', 'a photo of a small {}.', 
        'there is a {} in the scene.', 'there is the {} in the scene.',
        'this is a {} in the scene.', 'this is the {} in the scene.', 
        'this is one {} in the scene.'
    ]

    text_features = []
    loaded_files = []  # 记录成功加载的文件
    total_prompts = {  # 记录每个类别加载的prompt数量
        'chatgpt3.5': {'normal': 0, 'abnormal': 0},
        'winclip': {'normal': 0, 'abnormal': 0},
        'manual': {'normal': 0, 'abnormal': 0},
        'fixed_medical': {'normal': 0, 'abnormal': 0}
    }
    
    # 处理原有的prompt
    for i in range(len(prompt_state)):
        # i=0 时处理normal状态
        # i=1 时处理abnormal状态
        state_key = 'normal' if i == 0 else 'abnormal'
        
        prompted_state = [state.format(obj) for state in prompt_state[i]]
        
        prompted_sentence = []
        for s in prompted_state:
            for template in original_templates:
                prompted_sentence.append(template.format(s))
        
        base_prompt_count = len(prompted_sentence)
        
        # 如果启用JSON prompts，添加来自JSON文件的prompts
        if use_json_prompts:
            json_prompts = []
            
            # 遍历PROMPT_FILES字典中的所有文件
            for category, file_list in PROMPT_FILES.items():
                category_prompts = {'normal': 0, 'abnormal': 0}  # 记录当前类别加载的prompt数量
                
                if verbose and i == 0:
                    print(f"\n{'-'*20} {category} prompts {'-'*20}")
                
                for file_path in file_list:
                    try:
                        prompts = load_prompts(file_path)
                        # 只加载当前状态(normal或abnormal)的prompts
                        for prompt in prompts[state_key]['prompts']:
                            json_prompts.append(prompt.replace('{classname}', obj))
                        
                        # 添加调试信息，显示加载的提示数量
                        print(f"从 {file_path} 加载的 {state_key} 提示数量: {len(prompts[state_key]['prompts'])}")
                        
                        normal_count = len(prompts['normal']['prompts'])
                        abnormal_count = len(prompts['abnormal']['prompts'])
                        
                        if normal_count > 0 or abnormal_count > 0:
                            if verbose and i == 0:
                                print(f"✓ {file_path}:")
                                print(f"  - Normal prompts: {normal_count}")
                                print(f"  - Abnormal prompts: {abnormal_count}")
                                print(f"  - Total prompts: {normal_count + abnormal_count}")
                            
                            if file_path not in loaded_files:
                                loaded_files.append(file_path)
                            
                            category_prompts['normal'] += normal_count
                            category_prompts['abnormal'] += abnormal_count
                            
                    except Exception as e:
                        if verbose and i == 0:
                            print(f"✗ {file_path}: Failed - {str(e)}")
                        continue
                
                total_prompts[category]['normal'] += category_prompts['normal']
                total_prompts[category]['abnormal'] += category_prompts['abnormal']
            
            # 将所有JSON prompts添加到prompted_sentence
            prompted_sentence.extend(json_prompts)

            if verbose and i == 0:
                print(f"\n{'='*50}")
                print("Loading Summary:")
                print(f"{'='*50}")
                print(f"Base prompts: {base_prompt_count}")
                for category, counts in total_prompts.items():
                    print(f"{category} prompts:")
                    print(f"  - Normal: {counts['normal']}")
                    print(f"  - Abnormal: {counts['abnormal']}")
                    print(f"  - Total: {counts['normal'] + counts['abnormal']}")
                print(f"Total JSON files loaded: {len(loaded_files)}")
                print(f"Total prompts: {len(prompted_sentence)}")
                print(f"{'='*50}\n")

        # 转换为tensor并编码
        prompted_sentence = tokenize(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        text_features.append(class_embedding)

    text_features = torch.stack(text_features, dim=1).to(device)
    return text_features


def cos_sim(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])

def get_translation_mat(a, b):
    return torch.tensor([[1, 0, a],
                         [0, 1, b]])

def rot_img(x, theta):
    dtype =  torch.FloatTensor
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    return x

def translation_img(x, a, b):
    dtype =  torch.FloatTensor
    rot_mat = get_translation_mat(a, b)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid, padding_mode="reflection")
    return x

def hflip_img(x):
    x = K.geometry.transform.hflip(x)
    return x

def vflip_img(x):
    x = K.geometry.transform.vflip(x)
    return x

def rot90_img(x,k):
    # k is 0,1,2,3
    degreesarr = [0., 90., 180., 270., 360]
    degrees = torch.tensor(degreesarr[k])
    x = K.geometry.transform.rotate(x, angle = degrees, padding_mode='reflection')
    return x


def augment(fewshot_img, fewshot_mask=None):

    augment_fewshot_img = fewshot_img

    if fewshot_mask is not None:
        augment_fewshot_mask = fewshot_mask

        # rotate img
        for angle in [-np.pi/8, -np.pi/16, np.pi/16, np.pi/8]:
            rotate_img = rot_img(fewshot_img, angle)
            augment_fewshot_img = torch.cat([augment_fewshot_img, rotate_img], dim=0)

            rotate_mask = rot_img(fewshot_mask, angle)
            augment_fewshot_mask = torch.cat([augment_fewshot_mask, rotate_mask], dim=0)
        # translate img
        for a,b in [(0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
            trans_img = translation_img(fewshot_img, a, b)
            augment_fewshot_img = torch.cat([augment_fewshot_img, trans_img], dim=0)

            trans_mask = translation_img(fewshot_mask, a, b)
            augment_fewshot_mask = torch.cat([augment_fewshot_mask, trans_mask], dim=0)

        # hflip img
        flipped_img = hflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)
        flipped_mask = hflip_img(fewshot_mask)
        augment_fewshot_mask = torch.cat([augment_fewshot_mask, flipped_mask], dim=0)

        # vflip img
        flipped_img = vflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)

        flipped_mask = vflip_img(fewshot_mask)
        augment_fewshot_mask = torch.cat([augment_fewshot_mask, flipped_mask], dim=0)

    else:
        # rotate img
        for angle in [-np.pi/8, -np.pi/16, np.pi/16, np.pi/8]:
            rotate_img = rot_img(fewshot_img, angle)
            augment_fewshot_img = torch.cat([augment_fewshot_img, rotate_img], dim=0)

        # translate img
        for a,b in [(0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
            trans_img = translation_img(fewshot_img, a, b)
            augment_fewshot_img = torch.cat([augment_fewshot_img, trans_img], dim=0)

        # hflip img
        flipped_img = hflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)

        # vflip img
        flipped_img = vflip_img(fewshot_img)
        augment_fewshot_img = torch.cat([augment_fewshot_img, flipped_img], dim=0)

        B, _, H, W = augment_fewshot_img.shape
        augment_fewshot_mask = torch.zeros([B, 1, H, W])
    
    return augment_fewshot_img, augment_fewshot_mask

