import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(ROOT_DIR.as_posix())
from ram.models import ram, ram_plus


class Wrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        label_embed = torch.nn.functional.relu(self.model.wordvec_proj(self.model.label_embed))

        image_embeds = self.model.image_proj(self.model.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # recognized image tags using image-tag recogntiion decoder
        image_spatial_embeds = image_embeds[:, 1:, :]

        bs = image_spatial_embeds.shape[0]
        label_embed = label_embed.unsqueeze(0).repeat(bs, 1, 1)
        tagging_embed = self.model.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode="tagging",
        )

        logits = self.model.fc(tagging_embed[0]).squeeze(-1)

        targets = torch.where(
            torch.sigmoid(logits) > self.model.class_threshold.to(image.device),
            torch.tensor(1.0).to(image.device),
            torch.zeros(self.model.num_class).to(image.device),
        )

        return targets


class WrapperPlus(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image_embeds = self.model.image_proj(self.model.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        image_cls_embeds = image_embeds[:, 0, :]
        image_spatial_embeds = image_embeds[:, 1:, :]

        bs = image_spatial_embeds.shape[0]

        des_per_class = int(self.model.label_embed.shape[0] / self.model.num_class)

        image_cls_embeds = image_cls_embeds / image_cls_embeds.norm(dim=-1, keepdim=True)
        reweight_scale = self.model.reweight_scale.exp()
        logits_per_image = reweight_scale * image_cls_embeds @ self.model.label_embed.t()
        logits_per_image = logits_per_image.view(bs, -1, des_per_class)

        weight_normalized = F.softmax(logits_per_image, dim=2)
        label_embed_reweight = torch.empty(bs, self.model.num_class, 512).to(image.device).to(image.dtype)

        for i in range(bs):
            # 这里对 value_ori 进行 reshape，然后使用 broadcasting
            reshaped_value = self.model.label_embed.view(-1, des_per_class, 512)
            product = weight_normalized[i].unsqueeze(-1) * reshaped_value
            label_embed_reweight[i] = product.sum(dim=1)

        label_embed = torch.nn.functional.relu(self.model.wordvec_proj(label_embed_reweight))

        # recognized image tags using alignment decoder
        tagging_embed = self.model.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode="tagging",
        )

        logits = self.model.fc(tagging_embed[0]).squeeze(-1)

        targets = torch.where(
            torch.sigmoid(logits) > self.model.class_threshold.to(image.device),
            torch.tensor(1.0).to(image.device),
            torch.zeros(self.model.num_class).to(image.device),
        )

        return targets


def export(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    assert args.type in ["ram", "ram_plus"]
    if args.type == "ram":
        file_path = args.output_dir + "/ram.onnx"
        model = Wrapper(ram(pretrained=args.pretrained, image_size=384, vit="swin_l"))
    else:
        file_path = args.output_dir + "/ram_plus.onnx"
        model = WrapperPlus(ram_plus(pretrained=args.pretrained, image_size=384, vit="swin_l"))

    model.eval()
    model = model.to(device)

    # Export model
    image = torch.randn(1, 3, 384, 384).to(device)

    torch.onnx.export(
        model,
        image,
        file_path,
        input_names=["image"],
        output_names=["tag"],
        opset_version=16,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tag2Text inferece for tagging and captioning")
    parser.add_argument("--pretrained", "-p", type=str, required=True, help="path to pretrained model")
    parser.add_argument("--output-dir", "-o", type=str, required=True, help="path to output directory")
    parser.add_argument("--type", "-t", type=str, default="ram", help="type of model to export")
    args = parser.parse_args()

    export(args)
