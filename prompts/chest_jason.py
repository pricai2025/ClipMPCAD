import json
import itertools
import os

# 定义常量
CHEXPERT_TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}


def generate_all_combinations(disease_data):
    combinations = []
    severity_options = disease_data.get("severity", [""])
    subtype_options = disease_data.get("subtype", [""])
    location_options = disease_data.get("location", [""])

    for severity, subtype, location in itertools.product(severity_options, subtype_options, location_options):
        description = " ".join(filter(None, [severity, subtype, location])).strip()
        if description:
            combinations.append(description)

    return combinations


def generate_prompts():
    abnormal_prompts = set()

    # 为每个疾病生成所有可能的描述模板
    templates = [
        "a chest X-ray showing {combo}",
        "a medical scan revealing {combo}",
        "a diagnostic image demonstrating {combo}",
        "an X-ray indicating {combo}",
        "radiographic evidence of {combo}",
        "a chest radiograph displaying {combo}",
        "a medical image showing {combo}",
        "an X-ray examination revealing {combo}",
        "a diagnostic study demonstrating {combo}",
        "a chest examination showing {combo}",
        "an imaging study revealing {combo}",
        "a radiological examination demonstrating {combo}",
        "a thoracic image showing {combo}",
        "a chest view revealing {combo}",
        "a medical examination indicating {combo}"
    ]

    # 生成异常描述
    for disease, prompts in CHEXPERT_CLASS_PROMPTS.items():
        combinations = generate_all_combinations(prompts)
        for combo in combinations:
            for template in templates:
                abnormal_prompts.add(template.format(combo=combo))

    # 生成正常描述
    normal_prompts = set()
    normal_templates = [
        "a normal chest X-ray without evidence of {condition}",
        "a clear chest radiograph showing no signs of {condition}",
        "a medical scan demonstrating normal appearance without {condition}",
        "an X-ray showing normal findings without {condition}",
        "a diagnostic image revealing no evidence of {condition}",
        "a chest examination showing normal findings without {condition}",
        "a radiograph demonstrating absence of {condition}",
        "a normal medical scan without {condition}",
        "a chest study showing no indication of {condition}",
        "an imaging study revealing normal appearance without {condition}",
        "a thoracic examination showing normal findings without {condition}",
        "a radiological study demonstrating no signs of {condition}",
        "a medical image showing clear findings without {condition}",
        "a diagnostic study revealing normal structures without {condition}",
        "a chest view showing no abnormalities suggesting {condition}"
    ]

    for task in CHEXPERT_TASKS:
        for template in normal_templates:
            normal_prompts.add(template.format(condition=task.lower()))

    prompt_data = {
        "abnormal": {
            "instruction": "Comprehensive chest X-ray abnormality description prompts",
            "prompts": sorted(list(abnormal_prompts))
        },
        "normal": {
            "instruction": "Comprehensive chest X-ray normal description prompts",
            "prompts": sorted(list(normal_prompts))
        }
    }

    return prompt_data


# 生成提示数据
prompt_data = generate_prompts()

# 保存到文件
with open('flexible_medical_prompts.json', 'w', encoding='utf-8') as f:
    json.dump(prompt_data, f, indent=2, ensure_ascii=False)

# 打印统计信息
print(f"Generated {len(prompt_data['abnormal']['prompts'])} abnormal prompts")
print(f"Generated {len(prompt_data['normal']['prompts'])} normal prompts")
print(f"Total prompts: {len(prompt_data['abnormal']['prompts']) + len(prompt_data['normal']['prompts'])}")