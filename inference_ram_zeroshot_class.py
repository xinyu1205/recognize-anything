'''
 * The Recognize Anything Model (RAM) inference on zeroshot classes
 * Written by Xinyu Huang
'''
import argparse
import numpy as np
import random

import torch
import torchvision.transforms as transforms

from PIL import Image
from models.tag2text import ram

from models.zs_utils import build_zeroshot_label_embedding
from torch import nn

parser = argparse.ArgumentParser(
    description='Tag2Text inferece for tagging and captioning')
parser.add_argument('--image',
                    metavar='DIR',
                    help='path to dataset',
                    default='images/zeroshot_example.jpg')
parser.add_argument('--pretrained',
                    metavar='DIR',
                    help='path to pretrained model',
                    default='pretrained/tag2text_swin_14m.pth')
parser.add_argument('--image-size',
                    default=384,
                    type=int,
                    metavar='N',
                    help='input image size (default: 448)')

def inference(image, model):

    with torch.no_grad():
        tags = model.generate_tag_zeroshot(image)

    return tags[0]


if __name__ == "__main__":

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(), normalize
    ])

    #######load model
    model = ram(pretrained=args.pretrained,
                             image_size=args.image_size,
                             vit='swin_l')
    
    #######set zero shot interference
    zeroshot_label_embedding, zeroshot_categories = build_zeroshot_label_embedding()

    model.tag_list = np.array(zeroshot_categories)
    
    model.label_embed = nn.Parameter(zeroshot_label_embedding.float())

    model.num_class = len(zeroshot_categories)
    # the threshold for zero shot categories is often lower
    model.class_threshold = torch.ones(model.num_class) * 0.5
    #######

    model.eval()

    model = model.to(device)
    raw_image = Image.open(args.image).convert("RGB").resize(
        (args.image_size, args.image_size))
    image = transform(raw_image).unsqueeze(0).to(device)

    res = inference(image, model)
    print("Image Tags: ", res)
