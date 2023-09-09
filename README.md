# <font size=8> :label: Recognize Anything & Tag2Text </font>

[![Web Demo](https://img.shields.io/badge/🤗-HuggingFace%20Space-cyan.svg)](https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mhd-medfa/recognize-anything/blob/main/recognize_anything_demo.ipynb)

Official PyTorch Implementation of <a href="https://recognize-anything.github.io/">Recognize Anything: A Strong Image Tagging Model </a> and <a href="https://tag2text.github.io/">Tag2Text: Guiding Vision-Language Model via Image Tagging</a>.

- **Recognize Anything Model(RAM)** is an image tagging model, which can recognize any common category with high accuracy.
- **Tag2Text** is a vision-language model guided by tagging, which can support caption, retrieval and tagging.

<!-- Welcome to try our [RAM & Tag2Text web Demo! 🤗](https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text) -->

Both Tag2Text and RAM exihibit strong recognition ability. 
We have combined Tag2Text and RAM with localization models (Grounding-DINO and SAM) and developed a strong visual semantic analysis pipeline in the [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) project.

![](./images/ram_grounded_sam.jpg)



## :bulb: Highlight of RAM
RAM is a strong image tagging model, which can recognize any common category with high accuracy.
- **Strong and general.** RAM exhibits exceptional image tagging capabilities with powerful zero-shot generalization;
    - RAM showcases impressive zero-shot performance, significantly outperforming CLIP and BLIP.
    - RAM even surpasses the fully supervised manners (ML-Decoder).
    - RAM exhibits competitive performance with the Google tagging API.
- **Reproducible and affordable.** RAM requires Low reproduction cost with open-source and annotation-free dataset;
- **Flexible and versatile.** RAM offers remarkable flexibility, catering to various application scenarios.


<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="images/experiment_comparison.png" align="center" width="800" ></td>
  </tr>
  <p align="center">(Green color means fully supervised learning and Blue color means zero-shot performance.)</p>
</table>
</p>

<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="images/tagging_results.jpg" align="center" width="800" ></td>
  </tr>
</table>
</p>

RAM significantly improves the tagging ability based on the Tag2text framework.
- **Accuracy.** RAM utilizes a **data engine** to **generate** additional annotations and **clean** incorrect ones,  **higher accuracy** compared to Tag2Text.
- **Scope.** RAM upgrades the number of fixed tags from  3,400+ to **[6,400+](./ram/data/ram_tag_list.txt)** (synonymous reduction to 4,500+ different semantic tags), covering **more valuable categories**.
  Moreover, RAM is equipped with **open-set capability**, feasible to recognize tags not seen during training

## :sunrise: Highlight of Tag2text
Tag2Text is an efficient and controllable vision-language model with tagging guidance.
- **Tagging.** Tag2Text recognizes **[3,400+](./ram/data/tag_list.txt)** commonly human-used categories without manual annotations.
- **Captioning.** Tag2Text integrates **tags information** into text generation as the **guiding elements**, resulting in **more controllable and comprehensive descriptions**. 
- **Retrieval.** Tag2Text provides **tags** as **additional visible alignment indicators** for image-text retrieval. 

<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="images/tag2text_framework.png" align="center" width="800" ></td>
  </tr>
</table>
</p>
</details>


<!-- ## :sparkles: Highlight Projects with other Models
- [Tag2Text/RAM with Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) is trong and general pipeline for visual semantic analysis, which can automatically **recognize**, detect, and segment for an image!
- [Ask-Anything](https://github.com/OpenGVLab/Ask-Anything) is a multifunctional video question answering tool. Tag2Text provides powerful tagging and captioning capabilities as a fundamental component.
- [Prompt-can-anything](https://github.com/positive666/Prompt-Can-Anything) is a gradio web library that integrates SOTA multimodal large models, including Tag2text as the core model for graphic understanding -->


<!-- 
## :fire: News

- **`2023/06/08`**: We release the [Recognize Anything Model (RAM) Tag2Text web demo 🤗](https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text), checkpoints and inference code!
- **`2023/06/07`**: We release the [Recognize Anything Model (RAM)](https://recognize-anything.github.io/), a strong image tagging model!
- **`2023/06/05`**: Tag2Text is combined with [Prompt-can-anything](https://github.com/OpenGVLab/Ask-Anything).
- **`2023/05/20`**: Tag2Text is combined with [VideoChat](https://github.com/OpenGVLab/Ask-Anything).
- **`2023/04/20`**: We marry Tag2Text with with [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything).
- **`2023/04/10`**: Code and checkpoint is available Now!
- **`2023/03/14`**: [Tag2Text web demo 🤗](https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text) is available on Hugging Face Space!  -->





## :writing_hand: TODO 

- [x] Release Tag2Text demo.
- [x] Release checkpoints.
- [x] Release inference code.
- [x] Release RAM demo and checkpoints.
- [x] Release training codes.
- [ ] Release training datasets.



## :toolbox: Checkpoints

<!-- insert a table -->
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Backbone</th>
      <th>Data</th>
      <th>Illustration</th>
      <th>Checkpoint</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>RAM-14M</td>
      <td>Swin-Large</td>
      <td>COCO, VG, SBU, CC-3M, CC-12M</td>
      <td>Provide strong image tagging ability.</td>
      <td><a href="https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/ram_swin_large_14m.pth">Download  link</a></td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tag2Text-14M</td>
      <td>Swin-Base</td>
      <td>COCO, VG, SBU, CC-3M, CC-12M</td>
      <td>Support comprehensive captioning and tagging.</td>
      <td><a href="https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/blob/main/tag2text_swin_14m.pth">Download  link</a></td>
    </tr>
  </tbody>
</table>


## :running: Model Inference

### **Setting Up** ###

1. Install recognize-anything as a package:

```bash
pip install git+https://github.com/xinyu1205/recognize-anything.git
```

Then the RAM and Tag2Text model can be imported in other projects:

```python
from ram.models import ram, tag2text
```

### **RAM Inference** ##

Get the English and Chinese outputs of the images:
<pre/>
python inference_ram.py  --image images/demo/demo1.jpg \
--pretrained pretrained/ram_swin_large_14m.pth
</pre>


### **RAM Inference on Unseen Categories (Open-Set)** ##

Firstly, custom recognition categories in [build_openset_label_embedding](./ram/utils/openset_utils.py), then get the tags of the images:
<pre/>
python inference_ram_openset.py  --image images/openset_example.jpg \
--pretrained pretrained/ram_swin_large_14m.pth
</pre>


### **Tag2Text Inference** ##

Get the tagging and captioning results:
<pre/>
python inference_tag2text.py  --image images/demo/demo1.jpg \
--pretrained pretrained/tag2text_swin_14m.pth
</pre>
Or get the tagging and sepcifed captioning results (optional):
<pre/>python inference_tag2text.py  --image images/demo/demo1.jpg \
--pretrained pretrained/tag2text_swin_14m.pth \
--specified-tags "cloud,sky"</pre>


### **Batch Inference and Evaluation** ##
We release two datasets `OpenImages-common` (214 seen classes) and `OpenImages-rare` (200 unseen classes). Copy or sym-link test images of [OpenImages v6](https://storage.googleapis.com/openimages/web/download_v6.html) to `datasets/openimages_common_214/imgs/` and `datasets/openimages_rare_200/imgs`.

To evaluate RAM on `OpenImages-common`:

```bash
python batch_inference.py \
  --model-type ram \
  --checkpoint pretrained/ram_swin_large_14m.pth \
  --dataset openimages_common_214 \
  --output-dir outputs/ram
```

To evaluate RAM open-set capability on `OpenImages-rare`:

```bash
python batch_inference.py \
  --model-type ram \
  --checkpoint pretrained/ram_swin_large_14m.pth \
  --open-set \
  --dataset openimages_rare_200 \
  --output-dir outputs/ram_openset
```

To evaluate Tag2Text on `OpenImages-common`:

```bash
python batch_inference.py \
  --model-type tag2text \
  --checkpoint pretrained/tag2text_swin_14m.pth \
  --dataset openimages_common_214 \
  --output-dir outputs/tag2text
```

Please refer to `batch_inference.py` for more options. To get P/R in table 3 of our paper, pass `--threshold=0.86` for RAM and `--threshold=0.68` for Tag2Text.

To batch inference custom images, you can set up you own datasets following the given two datasets.


## :golfing: Model Training/Finetuning


### **Tag2Text** ##
At present, we can only open source [the forward function of Tag2Text](./ram/models/tag2text.py#L141) as much as possible.
To train/finetune Tag2Text on a custom dataset, you can refer to the complete training codebase of [BLIP](https://github.com/salesforce/BLIP/tree/main) and make the following modifications:
1. Replace the "models/blip.py" file with the current "[tag2text.py](./ram/models/tag2text.py)" model file;
2. Load additional tags based on the original dataloader.

### **RAM** ##

The training code of RAM cannot be open-sourced temporarily as it is in the company's process.



## :black_nib: Citation
If you find our work to be useful for your research, please consider citing.

```
@article{zhang2023recognize,
  title={Recognize Anything: A Strong Image Tagging Model},
  author={Zhang, Youcai and Huang, Xinyu and Ma, Jinyu and Li, Zhaoyang and Luo, Zhaochuan and Xie, Yanchun and Qin, Yuzhuo and Luo, Tong and Li, Yaqian and Liu, Shilong and others},
  journal={arXiv preprint arXiv:2306.03514},
  year={2023}
}

@article{huang2023tag2text,
  title={Tag2Text: Guiding Vision-Language Model via Image Tagging},
  author={Huang, Xinyu and Zhang, Youcai and Ma, Jinyu and Tian, Weiwei and Feng, Rui and Zhang, Yuejie and Li, Yaqian and Guo, Yandong and Zhang, Lei},
  journal={arXiv preprint arXiv:2303.05657},
  year={2023}
}
```

## :hearts: Acknowledgements
This work is done with the help of the amazing code base of [BLIP](https://github.com/salesforce/BLIP), thanks very much!

We want to thank @Cheng Rui @Shilong Liu @Ren Tianhe for their help in [marrying RAM/Tag2Text with Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything).

We also want to thank [Ask-Anything](https://github.com/OpenGVLab/Ask-Anything), [Prompt-can-anything](https://github.com/positive666/Prompt-Can-Anything) for  combining RAM/Tag2Text, which greatly expands the application boundaries of RAM/Tag2Text.
