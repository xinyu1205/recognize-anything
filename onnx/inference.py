import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parents[1]


def infer(args):
    # Load the model
    sess = ort.InferenceSession(args.pretrained)

    # Load the image
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess the image
    image = cv2.resize(image, (384, 384))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = image.astype(np.float32) / 255
    image = (image - mean) / std
    image = image.transpose(2, 0, 1).astype(np.float32)
    image = np.expand_dims(image, axis=0)

    # Load the tag list
    tag_list_path = ROOT_DIR / "ram/data/ram_tag_list.txt"
    with open(tag_list_path) as f:
        tag_list = f.read().splitlines()
    tag_list = np.array(tag_list)

    # Inference
    tag = sess.run(None, {sess.get_inputs()[0].name: image})[0][0]
    tokens = tag_list[tag == 1]
    print(tag)
    print("Image Tags: ", tokens.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tag2Text inferece for tagging and captioning")
    parser.add_argument("--pretrained", "-p", type=str, required=True, help="path to pretrained model")
    parser.add_argument("--image", "-i", type=str, required=True, help="path to image")
    args = parser.parse_args()

    infer(args)
