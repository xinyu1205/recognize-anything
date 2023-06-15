from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Union

from models import run_ram, run_ram_openset
from models.util.calculate_map import get_mAP
from models.util.calculate_pr import get_PR


def parse_args():
    parser = ArgumentParser()
    # model
    parser.add_argument("--backbone", type=str, default="swin_l",
                        choices=("swin_l", "swin_b"))
    parser.add_argument("--checkpoint", type=str,
                        default="pretrained/ram_swin_large_14m.pth")
    parser.add_argument("--open-set", action="store_true")

    # data
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dataset", type=str)
    group.add_argument("--image-path", type=str,
                       help="single image file path")
    parser.add_argument("--input-size", type=int, default=384)

    # threshold
    parser.add_argument(
        "--threshold", type=str, default="best",
        help=(
            "'paper': Thresholds used in our paper."
            "'best': Tuned thresholds, used in our HuggingFace demo."
            "Any float between 0~1: custom threshold."
        )
    )

    # miscellaneous
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    if args.dataset is None and args.image_path is None:
        raise ValueError("specify a dataset or an image file path")

    if args.open_set and args.image_path is not None:
        raise ValueError(
            "if you want open-set inference, please customize a dataset "
            "(must include tag list) following readme"
        )

    return args


def load_dataset(dataset: str) -> Dict[str, List]:
    dataset_root = str(Path(__file__).parent / "data" / dataset)

    taglist_file = dataset_root + f"/{dataset}_taglist.txt"
    if Path(taglist_file).is_file():
        with open(taglist_file, "r", encoding="utf-8") as f:
            taglist = [line.strip() for line in f]
        assert (
            all(tag for tag in taglist)
            and len(taglist) == int(dataset.split("_")[-1])
        )
    else:
        taglist = None

    annot_file = dataset_root + f"/{dataset}.txt"
    img_root = dataset_root + "/imgs"
    with open(annot_file, "r", encoding="utf-8") as f:
        imagelist = [img_root + "/" + line.strip().split(",")[0] for line in f]

    return {
        "taglist": taglist,
        "imagelist": imagelist,
        "annot_file": annot_file,
        "img_root": img_root
    }


def parse_threshold(
    arg_thre: str, openset: bool, taglist: List[str]
) -> Union[float, List[float]]:
    if arg_thre == "paper":
        if not openset:
            threshold = 0.86
        else:
            threshold = 0.576
    elif arg_thre == "best":
        if not openset:
            thre_file = str(
                Path(__file__).parent / "data/ram_tag_list_threshold.py")
            with open(thre_file, "r", encoding="utf-8") as f:
                threshold = eval(f.read().split("=")[-1])
            tag2thre = dict(zip(run_ram.load_model_taglist(), threshold))
            threshold = [tag2thre[tag] for tag in taglist]
        else:
            threshold = 0.5
    else:
        threshold = float(arg_thre)
    return threshold


def output_args(args, file: str) -> None:
    with open(file, "w", encoding="utf-8") as f:
        print("****************")
        f.write("****************\n")
        for key in (
            "backbone", "checkpoint", "open_set",
            "dataset", "image_path", "input_size",
            "threshold",
            "output_dir", "batch_size", "num_workers"
        ):
            s = f"{key}: {getattr(args, key)}"
            print(s)
            f.write(s + "\n")
        print("****************")
        f.write("****************\n")


def gen_pred_file(
    imagelist: List[str],
    tags: List[List[str]],
    img_root: str,
    pred_file: str
) -> None:
    with open(pred_file, "w", encoding="utf-8") as f:
        for image, tag in zip(imagelist, tags):
            s = str(Path(image).relative_to(img_root))
            if tag:
                s = s + "," + ",".join(tag)
            f.write(s + "\n")


if __name__ == "__main__":
    args = parse_args()

    # set up output paths
    output_dir = args.output_dir
    cache_dir = output_dir + "/cache"
    Path(cache_dir).mkdir(exist_ok=True, parents=True)
    pred_file = output_dir + f"/pred_th_{args.threshold}.txt"
    pr_file = output_dir + f"/pr_th_{args.threshold}.txt"
    ap_file = output_dir + "/ap.txt"
    summary_file = output_dir + f"/summary_th_{args.threshold}.txt"
    output_args(args, summary_file)

    # prepare data
    if args.dataset is not None:
        dataset = load_dataset(args.dataset)
        imagelist = dataset["imagelist"]
        taglist = dataset["taglist"]
        annot_file = dataset["annot_file"]
        img_root = dataset["img_root"]
    else:
        imagelist = [args.image_path]
        taglist = annot_file = None
        img_root = ""
    if taglist is None:
        taglist = run_ram.load_model_taglist()

    # set up threshold(s)
    threshold = parse_threshold(args.threshold, args.open_set, taglist)

    # run inference
    run_func = run_ram.run if not args.open_set else run_ram_openset.run
    results = run_func(
        backbone=args.backbone,
        checkpoint=args.checkpoint,
        imagelist=imagelist,
        input_size=args.input_size,
        taglist=taglist,
        threshold=threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=cache_dir
    )
    tags = results["tags"]
    logits = results["logits"]

    # generate text predicion file
    gen_pred_file(imagelist, tags, img_root, pred_file)

    # evaluate or print result
    if args.dataset is not None:  # evaluate
        CP, CR = get_PR(pred_file, annot_file, taglist, pr_file)
        mAP = get_mAP(logits, annot_file, taglist, ap_file)
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"CP: {CP}\n")
            f.write(f"CR: {CR}\n")
            f.write(f"mAP: {mAP}\n")
    else:  # print result of the image
        print(f"tags: {' | '.join(tags[0])}")
        if not args.open_set:
            en2ch = dict(zip(taglist, run_ram.load_chinese_model_taglist()))
            print(f"Chinese tags: {' | '.join([en2ch[en] for en in tags[0]])}")
