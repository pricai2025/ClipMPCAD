# ClipMPCAD

This is an official implementation of “Few-Shot Anomaly Detection with LLM-Guided Prompts and Multi-Attention Fusion” with PyTorch.

[Paper Link](https://arxiv.org/abs/2403.12570)



<center><img src="images/clipmpcad.png "width="60%"></center>

**Abstract**:   Abstract. Anomaly detection in medical and industrial images often faces challenges such as limited data availability and strict privacy constraints, which hinder large-scale model training. Although pre-trained large vision-language models (VLMs) offer promise in few-shot anomaly detection (FSAD), their effectiveness is limited by weak cross-modal
 alignment, insufficient domain-specific supervision, and poor sensitiv
ity to fine-grained local anomalies. To address these issues, we pro
pose ClipMPCAD, a CLIP-based framework for cross-domain FSAD
 that integrates large language model (LLM)-guided prompts and multi
attention mechanisms. Specifically, we introduce: (1) a Multi-Attention
 Driven Feature Fusion (MADFF) module that enhances spatial-frequency
 awareness and channel-level attention for precise localization; and (2) a
 Multi-level Semantic Decoder (M-Decoder) combined with Professional
domain Prompts (P-Prompts)—LLM-generated, domain-adaptive tex
tual embeddings that guide hierarchical visual-text alignment. Experi
ments on nine diverse datasets demonstrate the effectiveness of ClipM
PCAD, achieving average classification & segmentation accuracies of
 90.78% and 98.67% on medical data and 93.11% and 97.54% on industrial
 data, setting new benchmarks in cross-domain FSAD without additional
 fine-tuning.

**Keywords**: Few-Shot, Large Vision-Language Model, Anomaly Classification and Segmentation

<center><img src="images/pipeline.png "width="80%"></center>

## Get Started

### Environment
- python >= 3.8.5
- pytorch >= 1.10.0
- torchvision >= 0.11.1
- numpy >= 1.19.2
- scipy >= 1.5.2
- kornia >= 0.6.1
- pandas >= 1.1.3
- opencv-python >= 4.5.4
- pillow
- tqdm
- ftfy
- regex

### Device
Single NVIDIA A40 GPU

### Pretrained model
- CLIP: https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt

    Download and put it under `CLIP/ckpt` folder



### Medical Anomaly Detection Benchmark

1. (optional) Follow the [BMAD](https://github.com/DorisBao/BMAD) to apply for permission to download the relevant dataset. After extracting the data, reorganize the data benchmark according to the guidelines provided in our Appendix A.
2. We also provide the pre-processed benchmark. Please download the following dataset

    * Liver: https://drive.google.com/file/d/1xriF0uiwrgoPh01N6GlzE5zPi_OIJG1I/view?usp=sharing
    * Brain: https://drive.google.com/file/d/1YxcjcQqsPdkDO0rqIVHR5IJbqS9EIyoK/view?usp=sharing
    * HIS: https://drive.google.com/file/d/1hueVJZCFIZFHBLHFlv1OhqF8SFjUVHk6/view?usp=sharing
    * RESC: https://drive.google.com/file/d/1BqDbK-7OP5fUha5zvS2XIQl-_t8jhTpX/view?usp=sharing
    * OCT17: https://drive.google.com/file/d/1GqT0V3_3ivXPAuTn4WbMM6B9i0JQcSnM/view?usp=sharing
    * ChestXray: https://drive.google.com/file/d/15DhnBAk-h6TGLTUbNLxP8jCCDtwOHAjb/view?usp=sharing

3. Place it within the master directory `data` and unzip the dataset.

    ```
    tar -xvf Liver.tar.gz
    tar -xvf Brain.tar.gz
    tar -xvf Histopathology_AD.tar.gz
    tar -xvf Retina_RESC.tar.gz
    tar -xvf Retina_OCT2017.tar.gz
    tar -xvf Chest.tar.gz
    ```


### File Structure
After the preparation work, the whole project should have the following structure:

```
code
├─ ckpt
│  ├─ few-shot
│  └─ zero-shot
├─ CLIP
│  ├─ bpe_simple_vocab_16e6.txt.gz
│  ├─ ckpt
│  │  └─ ViT-L-14-336px.pt
│  ├─ clip.py
│  ├─ model.py
│  ├─ models.py
│  ├─ model_configs
│  │  └─ ViT-L-14-336.json
│  ├─ modified_resnet.py
│  ├─ openai.py
│  ├─ tokenizer.py
│  └─ transformer.py
├─ data
│  ├─ Brain_AD
│  │  ├─ valid
│  │  └─ test
│  ├─ ...
│  └─ Retina_RESC_AD
│     ├─ valid
│     └─ test
├─ dataset
│  ├─ fewshot_seed
│  │  ├─ Brain
│  │  ├─ ...
│  │  └─ Retina_RESC
│  ├─ medical_few.py
│  └─ medical_zero.py
├─ loss.py
├─ prompt.py
├─ readme.md
├─ train_few.py
├─ train_zero.py
└─ utils.py

```


### Quick Start

`python test_few.py --obj $target-object --shot $few-shot-number`

For example, to test on the Brain MRI with k=4, simply run:

`python test_few.py --obj Brain --shot 4`

### Training

`python train_few.py --obj $target-object --shot $few-shot-number`

For example, to train on the Brain MRI with k=4, simply run:

`python train_few.py --obj Brain --shot 4`


## Results

Results of zero-shot anomaly detection and localization:

<div style="text-align: center;">
<table>
<tr><td>AUC (%)</td> <td colspan="2">Detection</td> <td colspan="2">Localization</td></tr>
<tr><td>Zero-shot</td> <td>Paper</td> <td>Inplementation</td> <td>Paper</td> <td>Inplementation</td></tr>
<tr height='21' style='mso-height-source:userset;height:16pt' id='r0'>
<td height='21' class='x21' width='90' style='height:16pt;width:67.5pt;'>HIS</td>
<td class='x23' width='90' style='width:67.5pt;'>77.90</td>
<td class='x22' width='90' style='width:67.5pt;'>76.90</td>
<td class='x23' width='90' style='width:67.5pt;'>-</td>
<td class='x22' width='90' style='width:67.5pt;'>-</td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r2'>
<td height='21' class='x21' style='height:16pt;'>ChestXray</td>
<td class='x23'>71.11</td>
<td class='x22'>71.11</td>
<td class='x23'>-</td>
<td class='x22'>-</td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r8'>
<td height='21' class='x21' style='height:16pt;'>OCT17</td>
<td class='x23'>95.40</td>
<td class='x23'>95.40</td>
<td class='x23'>-</td>
<td class='x22'>-</td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r9'>
<td height='21' class='x21' style='height:16pt;'>BrainMRI</td>
<td class='x23'>78.63</td>
<td class='x22'>79.80</td>
<td class='x22'>90.27</td>
<td class='x23'>89.68</td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r10'>
<td height='21' class='x21' style='height:16pt;'>LiverCT</td>
<td class='x23'>76.24</td>
<td class='x22'>81.18</td>
<td class='x23'>97.85</td>
<td class='x22'>97.93</td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r11'>
<td height='21' class='x21' style='height:16pt;'>RESC</td>
<td class='x23'>83.31</td>
<td class='x22'>88.99</td>
<td class='x22'>92.05</td>
<td class='x22'>90.44</td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r11'>
<td height='21' class='x21' style='height:16pt;'>Average</td>
<td class='x23'>80.43</td>
<td class='x22'>82.23</td>
<td class='x22'>93.39</td>
<td class='x22'>92.68</td>
 </tr>
</table>
</div>

Results of few-shot anomaly detection and localization with k=4:

<div style="text-align: center;">
<table>
<tr><td>AUC (%)</td> <td colspan="2">Detection</td> <td colspan="2">Localization</td></tr>
<tr><td>4-shot</td> <td>Paper</td> <td>Inplementation</td> <td>Paper</td> <td>Inplementation</td></tr>
<tr height='21' style='mso-height-source:userset;height:16pt' id='r0'>
<td height='21' class='x21' width='90' style='height:16pt;width:67.5pt;'>HIS</td>
<td class='x23' width='90' style='width:67.5pt;'>82.71</td>
<td class='x22' width='90' style='width:67.5pt;'>82.71</td>
<td class='x23' width='90' style='width:67.5pt;'>-</td>
<td class='x22' width='90' style='width:67.5pt;'>-</td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r2'>
<td height='21' class='x21' style='height:16pt;'>ChestXray</td>
<td class='x23'>81.95</td>
<td class='x22'>81.95</td>
<td class='x23'>-</td>
<td class='x22'>-</td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r8'>
<td height='21' class='x21' style='height:16pt;'>OCT17</td>
<td class='x23'>99.38</td>
<td class='x23'>99.38</td>
<td class='x23'>-</td>
<td class='x22'>-</td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r9'>
<td height='21' class='x21' style='height:16pt;'>BrainMRI</td>
<td class='x23'>92.44</td>
<td class='x22'>92.31</td>
<td class='x22'>97.30</td>
<td class='x23'>97.30</td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r10'>
<td height='21' class='x21' style='height:16pt;'>LiverCT</td>
<td class='x23'>81.18</td>
<td class='x22'>81.18</td>
<td class='x23'>99.73</td>
<td class='x22'>99.69</td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r11'>
<td height='21' class='x21' style='height:16pt;'>RESC</td>
<td class='x23'>96.18</td>
<td class='x22'>96.18</td>
<td class='x22'>98.97</td>
<td class='x22'>98.97</td>
 </tr>
 <tr height='21' style='mso-height-source:userset;height:16pt' id='r11'>
<td height='21' class='x21' style='height:16pt;'>Average</td>
<td class='x23'>88.97</td>
<td class='x22'>88.95</td>
<td class='x22'>98.67</td>
<td class='x22'>98.65</td>
 </tr>
</table>
</div>


## Visualization
<center><img src="images/visualize.png "width="70%"></center>

## Acknowledgement
We borrow some codes from [OpenCLIP](https://github.com/mlfoundations/open_clip), and [April-GAN](https://github.com/ByChelsea/VAND-APRIL-GAN).

## Contact

If you have any problem with this code, please feel free to contact **huangchaoqin@sjtu.edu.cn** and **stillunnamed@sjtu.edu.cn**.

