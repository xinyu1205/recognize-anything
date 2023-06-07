# :label: Recognize Anything: A Strong Image Tagging Model & Tag2Text: Guiding Vision-Language Model via Image Tagging

Official PyTorch Implementation of the <a href="https://recognize-anything.github.io/">Recognize Anything Model (RAM)</a> and the <a href="https://tag2text.github.io/">Tag2Text Model</a>.

- RAM is a strong image tagging model, which can recognize any common category with high accuracy.
- Tag2Text is an efficient and controllable vision-language model with tagging guidance.


When combined with localization models ([Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)), Tag2Text and RAM form a strong and general pipeline for visual semantic analysis.

![](./images/ram_grounded_sam.jpg)

## :sun_with_face: Helpful Tutorial


- :apple:  [[Access RAM Homepage](https://recognize-anything.github.io/)]
- :grapes: [[Access Tag2Text Homepage](https://tag2text.github.io/)]
- :sunflower:  [[Read RAM arXiv Paper](https://arxiv.org/abs/2306.03514)]
- :rose: [[Read Tag2Text arXiv Paper](https://arxiv.org/abs/2303.05657)]
- :mushroom: [[Try our Tag2Text web Demo! 🤗](https://huggingface.co/spaces/xinyu1205/Tag2Text)]



## :bulb: Highlight
**Recognition and localization are two foundation computer vision tasks.**
- **The Segment Anything Model (SAM)** excels in **localization capabilities**, while it falls short when it comes to **recognition tasks**.
- **The Recognize Anything Model (RAM) and Tag2Text** exhibits **exceptional recognition abilities**, in terms of **both accuracy and scope**.

<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="images/localization_and_recognition.jpg" align="center" width="800" ></td>
  </tr>
</table>
</p>


<details close>
<summary><font size="4">
Tag2Text for Vision-Language Tasks.
</font></summary>

- **Tagging.** Without manual annotations, Tag2Text achieves **superior** image tag recognition ability of [**3,429**](./data/tag_list.txt) commonly human-used categories.
- **Efficient.** Tagging guidance effectively enhances the performance of vision-language models on both **generation-based** and **alignment-based** tasks.
- **Controllable.** Tag2Text permits users to input **desired tags**, providing the flexibility in composing corresponding texts based on the input tags.


<p align="center">
 <table class="tg">
  <tr>
    <td class="tg-c3ow"><img src="images/tag2text_framework.png" align="center" width="800" ></td>
  </tr>
</table>
</p>
</details>


<details close>
<summary><font size="4">
Advancements of RAM on Tag2Text.
</font></summary>

- **Accuracy.** RAM utilizes a data engine to generate additional annotations and clean incorrect ones, resulting higher accuracy compared to Tag2Text.
- **Scope.** Tag2Text recognizes 3,400+ fixed tags. RAM upgrades the number to 6,400+, covering more valuable categories. With open-set capability, RAM is feasible to recognize any common category.


</details>


## :sparkles: Highlight Projects with other Models
- [Tag2Text/RAM with Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) is trong and general pipeline for visual semantic analysis, which can automatically **recognize**, detect, and segment for an image!
- [Ask-Anything](https://github.com/OpenGVLab/Ask-Anything) is a multifunctional video question answering tool. Tag2Text provides powerful tagging and captioning capabilities as a fundamental component.
- [Prompt-can-anything](https://github.com/positive666/Prompt-Can-Anything) is a gradio web library that integrates SOTA multimodal large models, including Tag2text as the core model for graphic understanding




## :fire: News

- **`2023/06/07`**: We release the [Recognize Anything Model (RAM)](https://recognize-anything.github.io/), a strong image tagging model!
- **`2023/06/05`**: Tag2Text is combined with [Prompt-can-anything](https://github.com/OpenGVLab/Ask-Anything).
- **`2023/05/20`**: Tag2Text is combined with [VideoChat](https://github.com/OpenGVLab/Ask-Anything).
- **`2023/04/20`**: We marry Tag2Text with with [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything).
- **`2023/04/10`**: Code and checkpoint is available Now!
- **`2023/03/14`**: [Tag2Text web demo 🤗](https://huggingface.co/spaces/xinyu1205/Tag2Text) is available on Hugging Face Space! 






## :writing_hand: TODO 

- [x] Release Tag2Text demo.
- [x] Release checkpoints.
- [x] Release inference code.
- [ ] Release RAM demo and checkpoints (until June 14th at the latest).
- [ ] Release training codes (until August 1st at the latest).
- [ ] Release training datasets (until August 1st at the latest).



## :toolbox: Checkpoints

<!-- insert a table -->
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>Data</th>
      <th>Illustration</th>
      <th>Checkpoint</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Tag2Text-Swin</td>
      <td>Swin-Base</td>
      <td>COCO, VG, SBU, CC-3M, CC-12M</td>
      <td>Demo version with comprehensive captions.</td>
      <td><a href="https://huggingface.co/spaces/xinyu1205/Tag2Text/blob/main/tag2text_swin_14m.pth">Download  link</a></td>
    </tr>
  </tbody>
</table>


## :running: Tag2Text Inference

1. Install the dependencies, run:

<pre/>pip install -r requirements.txt</pre> 

2. Download Tag2Text pretrained checkpoints.

3. Get the tagging and captioning results:
<pre/>
python inference.py  --image images/1641173_2291260800.jpg \
--pretrained pretrained/tag2text_swin_14m.pth
</pre>
Or get the tagging and sepcifed captioning results (optional):
<pre/>python inference.py  --image images/1641173_2291260800.jpg \
--pretrained pretrained/tag2text_swin_14m.pth \
--specified-tags "cloud,sky"</pre>


## :black_nib: Citation
If you find our work to be useful for your research, please consider citing.

```
@misc{zhang2023recognize,
  title={Recognize Anything: A Strong Image Tagging Model}, 
  author={Youcai Zhang and Xinyu Huang and Jinyu Ma and Zhaoyang Li and Zhaochuan Luo and Yanchun Xie and Yuzhuo Qin and Tong Luo and Yaqian Li and Shilong Liu and Yandong Guo and Lei Zhang},
  year={2023},
  eprint={2306.03514},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
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

We also want to thank @Cheng Rui @Shilong Liu @Ren Tianhe for their help in [marrying Tag2Text with Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything).







