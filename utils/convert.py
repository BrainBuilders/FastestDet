#!/usr/bin/env python

import json
import random
import shutil
import sys
import argparse
from pathlib import Path


def write_categories(data, root: Path):
    cats = data["categories"]
    with open(root / "category.names", "wt") as f:
        for c in cats:
            f.write(f"{c['name']}\n")


def find_annotations(img, ann):
    img_anns = []
    for a in ann:
        if a["image_id"] == img["id"]:
            img_anns.append(a)

    labels = []
    img_width = img["width"]
    img_height = img["height"]
    for a in img_anns:
        x, y, w, h = a["bbox"]
        x += w / 2
        x /= img_width
        y += h / 2
        y /= img_height
        w /= img_width
        h /= img_height
        label = (a["category_id"] - 1, x, y, w, h)
        labels.append(label)

    return labels


def write_annotations(img, ann, txt):
    labels = find_annotations(img, ann)
    with open(str(txt), "wt") as f:
        for c, x, y, w, h in labels:
            f.write(f"{c} {x:0.5f} {y:0.5f} {w:0.5f} {h:0.5f}\n")


def main(argv) -> int:
    root = argv.output

    if root.exists():
        print(f"Directory {root} already exists")
        return 1

    root.mkdir(parents=True)
    train = root / "train"
    train.mkdir()
    val = root / "val"
    val.mkdir()

    with open(str(argv.cocojson), "rt") as f:
        data = json.load(f)

    write_categories(data, root)
    
    valid = []
    for imgdata in data["images"]:
        path = argv.cocojson.parent.parent / "images" / imgdata["file_name"]
        if path.is_file():
            valid.append(imgdata)
    n = len(valid)
    print(f"found {n} valid images")

    random.shuffle(valid)
    tn = round(n * argv.ratio)

    with open(root / "train.txt", "wt") as f:
        for imgdata in valid[:tn]:
            src = argv.cocojson.parent.parent / "images" / imgdata["file_name"]
            dst = train / imgdata["file_name"]
            txt = dst.with_suffix(".txt")
            write_annotations(imgdata, data["annotations"], txt)
            shutil.copy(str(src), str(dst))
            f.write(f"{dst}\n")

    with open(root / "val.txt", "wt") as f:
        for imgdata in valid[tn:]:
            src = argv.cocojson.parent.parent / "images" / imgdata["file_name"]
            dst = val / imgdata["file_name"]
            txt = dst.with_suffix(".txt")
            write_annotations(imgdata, data["annotations"], txt)
            shutil.copy(str(src), str(dst))
            f.write(f"{dst}\n")

    return 0


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
            "Convert coco dataset to darknet format", \
			formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("cocojson", type=Path,
            help="coco type json file")
    argparser.add_argument("output", type=Path,
            help="output directory")
    argparser.add_argument("--ratio", type=float, default=0.9,
            help="train/val ratio")
    argv = argparser.parse_args()
    sys.exit(main(argv))

