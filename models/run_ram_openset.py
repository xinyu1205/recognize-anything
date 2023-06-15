from pathlib import Path
from typing import Dict, List, Union

import torch
from PIL import Image, ImageFile, UnidentifiedImageError
from torch import Tensor
from torch.nn.functional import sigmoid
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from .openset_utils import build_openset_label_embedding
from .run_ram import model_forward
from .tag2text import ram
from .util.inference import inference

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    backbone: str,
    checkpoint: str,
    input_size: int,
    label_embed: Tensor
):
    model = ram(pretrained=checkpoint, image_size=input_size, vit=backbone)
    model.label_embed = torch.nn.Parameter(label_embed.float())
    return model.to(device).eval()


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

    # 用clip生成label_embedding
    label_embed_file = Path(cache_dir) / "label_embeds.pth"
    if label_embed_file.is_file():
        label_embeds = torch.load(str(label_embed_file)).to(device)
    else:
        label_embeds = build_openset_label_embedding(taglist, device)
        torch.save(label_embeds, str(label_embed_file))

    # inference
    logits_file = Path(cache_dir) / "logits.pth"
    if logits_file.is_file():
        logits = torch.load(str(logits_file))
    else:
        if model is None:
            model = load_model(backbone, checkpoint, input_size, label_embeds)

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
