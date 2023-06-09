# -*- coding: utf-8 -*-
import argparse
import json
import os
import time

import imghdr
import torch
import torchvision.transforms as transforms
from PIL import Image

from models.tag2text import tag2text_caption, ram


def parse_args():
    """
    This function parses command line arguments for a Tag2Text inference model.
    :return: The function `parse_args()` is returning the parsed arguments from the command line using
    the `argparse` module.
    """
    parser = argparse.ArgumentParser(
        description='Tag2Text inference for tagging and captioning')
    parser.add_argument('--image-dir',
                        metavar='DIR',
                        help='path to directory containing input images',
                        default='')
    parser.add_argument('--images',
                        metavar='IMAGE-LIST',
                        nargs='+',
                        help='list of space-separated image filenames',
                        default=[])
    parser.add_argument('--pretrained',
                        metavar='DIR',
                        help='path to pretrained model',
                        default='D:/work/Tag2Text/pretrained/tag2text_swin_14m.pth')
    parser.add_argument('--image-size',
                        default=384,
                        type=int,
                        metavar='N',
                        help='input image size (default: 448)')
    parser.add_argument('--thre',
                        default=0.68,
                        type=float,
                        metavar='N',
                        help='threshold value')
    parser.add_argument('--specified-tags',
                        default='None',
                        help='User input specified tags')
    parser.add_argument('--model-type',
                        default='tag2text',
                        help='Assignment model')
    parser.add_argument('--cache-path',
                        default='None',
                        help='cache model file path')

    return parser.parse_args()


def initialize_model(cache_path, pretrained, image_size, thre, model_type):
    """
    This function initializes a Tag2Text model based on specified and identified tags.
    :param cache_path: Cache model file path.
    :param pretrained: Path to the pre-trained model.
    :param image_size: Input image size.
    :param thre: Threshold value for tagging.
    :return: A pre-trained Tag2Text model.
    """

    # delete some tags that may disturb captioning
    # 127: "quarter"; 2961: "back", 3351: "two"; 3265: "three"; 3338: "four"; 3355: "five"; 3359: "one"
    delete_tag_index = [127, 2961, 3351, 3265, 3338, 3355, 3359]

    if model_type == 'tag2text':
        model = tag2text_caption(
            pretrained=pretrained,
            image_size=image_size,
            vit='swin_b',
            delete_tag_index=delete_tag_index
        )
    elif model_type == 'ram':
        model = ram(pretrained=pretrained,
                                image_size=image_size,
                                vit='swin_l')
        
    model.threshold = thre  # threshold for tagging
    model.eval()

    return model


def generate(model, image, input_tags=None, model_type='tag2text'):
    """
    This function generates tags and captions for an input image.
    :param model: The neural network model used for generating captions and predicting tags for an input
    image.
    :param image: The input image to generate tags and captions for.
    :param input_tags: The input tags used as hints for the model to generate captions for the input image.
    It is an optional parameter and can be set to None or left empty if no tag hint is required.
    :return: A tuple of predicted tags, input tags, and generated captions.
    """

    if input_tags in ('', 'none', 'None'):
        input_tags = None

    with torch.no_grad():
        if model_type == 'tag2text':
            caption, tag_predict = model.generate(image,
                                                tag_input=None,
                                                max_length=50,
                                                return_tag_predict=True)
        elif model_type == 'ram':
            tag_predict, _ = model.generate_tag(image)
            caption = [None]

    if input_tags is None:
        return tag_predict[0], None, caption[0]

    input_tag_list = [input_tags.replace(',', ' | ')]
    with torch.no_grad():
        if model_type == 'tag2text':
            caption, input_tags = model.generate(image,
                                                tag_input=input_tag_list,
                                                max_length=50,
                                                return_tag_predict=True)
        elif model_type == 'ram':
            tag_predict, _ = model.generate_tag(image)
            caption = [None]

    return tag_predict[0], input_tags[0], caption[0]


def inference(images_dir, image_list, model, image_size, input_tags=None, model_type='tag2text'):
    """
    This function takes a list of images or a directory containing images, a model, generates captions
    for the images, and optionally takes a list of input tags to generate captions with those tags.
    :param images_dir: A directory containing input images that the model will use to generate captions and
    potentially predict tags for.
    :param image_list: A list of input images the model will use to generate captions and potentially
    predict tags for.
    :param model: The neural network model used for generating captions and predicting tags for an input
    image.
    :param input_tags: The input tags are lists of strings that represent tags or sets of tags that are
    used as hints for the model to generate captions for the given images. It is an optional parameter and
    can be set to None or left empty if no tag hint is required, defaults to None.
    :return: A list of dictionaries, each containing predicted tags, input tags (if provided), and
    generated captions for a given input image.
    """
    
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), normalize
    ])

    if images_dir and os.path.isdir(images_dir):
        for filename in os.listdir(images_dir):
            filepath = os.path.join(images_dir, filename)
            if not imghdr.what(filepath):
                continue
            img = Image.open(filepath).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            res = generate(model, img_tensor, input_tags, model_type)
            results.append({
                "filepath": filepath,
                "model_identified_tags": res[0],
                "user_specified_tags": res[1],
                "image_caption": res[2]
            })
            print(results[-1])
    elif image_list and isinstance(image_list, list):
        for img_path in image_list:
            filepath = os.path.abspath(img_path)
            if not os.path.isfile(filepath) or not imghdr.what(filepath):
                continue
            img = Image.open(filepath).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            res = generate(model, img_tensor, input_tags, model_type)
            results.append({
                "filepath": img_path,
                "model_identified_tags": res[0],
                "user_specified_tags": res[1],
                "image_caption": res[2]
            })
            print(results[-1])

    return results

def main():
    """
    This function loads a pre-trained image captioning model, processes input images in a directory,
    and generates captions for each image based on specified and identified tags.
    """
    start_time = time.time()
    args = parse_args()

    # check if a list of images is provided
    images = args.images if args.images else None
    # initialize the model
    model = initialize_model(
        args.cache_path, args.pretrained, args.image_size, args.thre, args.model_type)

    # perform inference on images
    data = inference(args.image_dir, images, model,
                    args.image_size, input_tags=None, model_type=args.model_type)

    # output the results
    results = {
        "status": 0,
        "message": 'ok',
        "data": data
    }

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(
        f"Processed {len(results['data'])} images in {elapsed_time:.2f} seconds.")

    json_results = json.dumps(results, ensure_ascii=False, indent=2)
    print(json_results)

    f2 = open(f'{args.model_type}_results.json', 'w')
    f2.write(json_results)
    f2.close()

# 使用示例：
# 1. python batch_inference.py --pretrained pretrain/tag2text_swin_14m.pth --image-dir image_dir --model-type tag2text
# 2. python batch_inference.py --pretrained pretrain/ram_swin_large_14m.pth --image-dir image_dir --model-type ram


if __name__ == '__main__':
    main()