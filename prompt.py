TEMPLATES = ['a cropped photo of the {}.', 'a cropped photo of a {}.', 'a close-up photo of a {}.', 'a close-up photo of the {}.', \
    'a bright photo of a {}.', 'a bright photo of the {}.', 'a dark photo of the {}.', 'a dark photo of a {}.',\
        'a jpeg corrupted photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a blurry photo of the {}.', 'a blurry photo of a {}.',\
            'a photo of the {}', 'a photo of a {}', 'a photo of a small {}', 'a photo of the small {}', 'a photo of a large {}',\
                'a photo of the large {}', 'a photo of the {} for visual inspection.', 'a photo of a {} for visual inspection.',\
                    'a photo of the {} for anomaly detection.', 'a photo of a {} for anomaly detection.',
    # MVTec 特定的模板
    'a photo of a normal {}.', 'a photo of an abnormal {}.',
    'a photo of a defect-free {}.', 'a photo of a defective {}.',
    'a high quality {}.', 'a low quality {}.',
    'a good {}.', 'a bad {}.',
    'an industrial photo of a normal {}.', 'an industrial photo of a defective {}.',
    'a manufacturing photo of a good {}.', 'a manufacturing photo of a faulty {}.',
    'a quality control photo of a perfect {}.', 'a quality control photo of an imperfect {}.',
    'an inspection photo of an acceptable {}.', 'an inspection photo of an unacceptable {}.',
    # MPDD 特定的模板
    'a photo of a normal {}.', 
    'a photo of an abnormal {}.',
    'a photo of a defect-free {}.', 
    'a photo of a defective {}.',
    'a high quality {}.', 
    'a low quality {}.',
    'a good {}.', 
    'a bad {}.',
    'an industrial photo of a normal {}.', 
    'an industrial photo of a defective {}.',
    'a manufacturing photo of a good {}.', 
    'a manufacturing photo of a faulty {}.',
    'a quality control photo of a perfect {}.', 
    'a quality control photo of an imperfect {}.',
    'an inspection photo of an acceptable {}.', 
    'an inspection photo of an unacceptable {}.',
]


REAL_NAME = {
    # 原有的医疗数据集类别
    'Brain': 'Brain', 
    'Liver': 'Liver',
    'Retina_RESC': 'retinal OCT', 
    'Chest': 'Chest X-ray film', 
    'Retina_OCT2017': 'retinal OCT', 
    'Histopathology': 'histopathological image',
    
    # MVTec AD 数据集的类别
    'bottle': 'bottle with defects',
    'cable': 'cable with defects',
    'capsule': 'capsule with defects',
    'carpet': 'carpet with defects',
    'grid': 'grid with defects',
    'hazelnut': 'hazelnut with defects',
    'leather': 'leather with defects',
    'metal_nut': 'metal nut with defects',
    'pill': 'pill with defects',
    'screw': 'screw with defects',
    'tile': 'tile with defects',
    'toothbrush': 'toothbrush with defects',
    'transistor': 'transistor with defects',
    'wood': 'wood with defects',
    'zipper': 'zipper with defects',

    # ViSA 数据集的类别
    'candle': 'candle with defects',
    'capsules': 'capsules with defects',
    'cashew': 'cashew with defects',
    'chewinggum': 'chewing gum with defects',
    'fryum': 'fryum with defects',
    'macaroni1': 'macaroni type 1 with defects',
    'macaroni2': 'macaroni type 2 with defects',
    'pcb1': 'PCB type 1 with defects',
    'pcb2': 'PCB type 2 with defects',
    'pcb3': 'PCB type 3 with defects',
    'pcb4': 'PCB type 4 with defects',
    'pipe_fryum': 'pipe fryum with defects',
    # MPDD 数据集的类别 - 修改为单个字符串
    'bracket_black': 'black bracket with defects',
    'bracket_brown': 'brown bracket with defects',
    'bracket_white': 'white bracket with defects',
    'connector': 'connector with defects',
    'metal_plate': 'metal plate with defects',
    'tubes': 'tube with defects',
    # BTAD 数据集的类别
    '01': 'silicon steel with defects',  # 硅钢
    '02': 'transistor with defects',     # 晶体管
    '03': 'wood with defects',           # 木材
}

# TEMPLATES = ['a cropped photo of the {}.', 'a cropped photo of a {}.', 'a close-up photo of a {}.', 'a close-up photo of the {}.', \
#     'a bright photo of a {}.', 'a bright photo of the {}.', 'a dark photo of the {}.', 'a dark photo of a {}.',\
#         'a jpeg corrupted photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a blurry photo of the {}.', 'a blurry photo of a {}.',\
#             'a photo of the {}', 'a photo of a {}', 'a photo of a small {}', 'a photo of the small {}', 'a photo of a large {}',\
#                 'a photo of the large {}', 'a photo of the {} for visual inspection.', 'a photo of a {} for visual inspection.',\
#                     'a photo of the {} for anomaly detection.', 'a photo of a {} for anomaly detection.']
#
#
# REAL_NAME = {'Brain': 'Brain', 'Liver':'Liver','Retina_RESC':'retinal OCT', 'Chest':'Chest X-ray film', 'Retina_OCT2017':'retinal OCT', 'Histopathology':'histopathological image'}
#
