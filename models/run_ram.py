from pathlib import Path
from typing import Dict, List, Union

import torch
from PIL import Image, ImageFile, UnidentifiedImageError
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import relu, sigmoid
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from .tag2text import ram
from .util.inference import inference

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_taglist() -> List[str]:
    model_taglist_file = str(
        Path(__file__).parent.parent / "data/ram_tag_list.txt")
    with open(model_taglist_file, "r", encoding="utf-8") as f:
        model_taglist = [line.strip() for line in f]
    return model_taglist


def load_chinese_model_taglist() -> List[str]:
    chinese_model_taglist_file = str(
        Path(__file__).parent.parent / "data/ram_tag_list_chinese.txt")
    with open(chinese_model_taglist_file, "r", encoding="utf-8") as f:
        chinese_model_taglist = [line.strip() for line in f]
    return chinese_model_taglist


def load_model(
    backbone: str,
    checkpoint: str,
    input_size: int,
    taglist: List[str]
) -> Module:
    model = ram(pretrained=checkpoint, image_size=input_size, vit=backbone)

    # remove redundant embeddings to speed up
    model_taglist = load_model_taglist()
    model_tag2idxs = {tag: idx for idx, tag in enumerate(model_taglist)}
    idxs = torch.tensor([model_tag2idxs[tag] for tag in taglist])
    label_embed = model.label_embed[idxs, :]
    model.label_embed = torch.nn.Parameter(label_embed)

    return model.to(device).eval()


@torch.no_grad()
def model_forward(model: Module, imgs: Tensor) -> Tensor:
    label_embed = relu(
        model.wordvec_proj(model.label_embed)
    ).unsqueeze(0).repeat(imgs.shape[0], 1, 1)

    image_embeds = model.image_proj(model.visual_encoder(imgs))

    image_atts = torch.ones(
        image_embeds.size()[:-1], dtype=torch.long).to(imgs.device)

    tagging_embedding = model.tagging_head(
        encoder_embeds=label_embed,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=False,
        mode='tagging',
    )

    logits = model.fc(tagging_embedding[0]).squeeze(-1)
    return logits


def run(
    backbone: str,
    checkpoint: str,
    imagelist: List[str],
    input_size: int,
    taglist: List[str],
    threshold: Union[float, List[float]],
    batch_size: int,
    num_workers: int,
    cache_dir: str
) -> Dict:
    # model lazy init
    model = None

    # inference
    logits_file = Path(cache_dir) / "logits.pth"
    if logits_file.is_file():
        logits = torch.load(str(logits_file))
    else:
        if model is None:
            model = load_model(backbone, checkpoint, input_size, taglist)

        transforms = Compose([
            Resize((input_size, input_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        def image_processor(image_path):
            try:
                img = Image.open(image_path).convert("RGB")
            except (OSError, FileNotFoundError, UnidentifiedImageError):
                img = Image.new('RGB', (input_size, input_size), 0)
                print("Error loading image:", image_path)
            return transforms(img)

        logits = inference(
            model=lambda input: model_forward(model, input.to(device)),
            datalist=imagelist,
            data_processor=image_processor,
            output_processor=lambda out: sigmoid(out),
            to_cpu=False,
            batch_size=batch_size,
            num_workers=num_workers
        ).cpu()

        torch.save(logits, str(logits_file))

    # filtering with thresholds
    tags = []
    if isinstance(threshold, list):
        assert len(threshold) == len(taglist)
        thresholds = threshold
    else:
        assert isinstance(threshold, float)
        thresholds = [threshold] * len(taglist)
    for scores in logits.tolist():
        tags.append([
            taglist[i] for i, s in enumerate(scores) if s >= thresholds[i]
        ])

    #
    return {"tags": tags, "logits": logits}
