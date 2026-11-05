"""Detect resistors in an image with a YOLO or RF-DETR pose model.

This script processes input images or directories to identify and locate resistors
using a specified pose estimation model. It supports both YOLO and RF-DETR 
architectures and handles coordinate transformations between raw and canonical spaces.

Example Usage:
    uv run inference.py --model path/to/weights.pth --output ./results img1.jpg img2.jpg
"""

from helpers import keypoints_to_components, draw_keypoints, resize_width, remap_keypoints, draw_components

from breadboard.rectify import rectify, load_photo
from breadboard.netlist import Netlist

from typing import Literal

from rfdetr import RFDETRKeypointPreview
from ultralytics import YOLO

from pathlib import Path

import numpy as np

import argparse
from PIL import Image

import supervision as sv

CLASS_NAMES = ["resistor"]
PIN_KEYPOINT_MASKS = np.array([[1, 1]], dtype=bool)

def is_image(path):
    try:
        with Image.open(path) as im:
            # no idea what this actually does, all the source code seems to do nothing
            # its probably already verified if it passes open()
            im.verify() 
            return True
    except:
        pass
    return False

def get_image_files(paths):
    results = []
    for p in map(Path, paths):
        if p.is_dir():
            results.extend(f for f in sorted(p.iterdir()) if is_image(f))
        else:
            results.append(p)
    return results
    
def load_images(image_paths: list[Path], should_rectify: bool):
    for path in image_paths:
        image = load_photo(path)
        if should_rectify:
            norm = rectify(image)
            if norm is None or not norm.ok:
                print(f"Failed to rectify image {path.stem}, continuing...")
            else:
                yield norm.canonical, path
        else:
            yield image, path

def predict_yolo(model: YOLO, image: np.ndarray, **kwargs):
    # yolo assumes numpy images are BGR
    results = model.predict(image[..., ::-1], **kwargs)

    kp = sv.KeyPoints.from_ultralytics(results[0])
    det = sv.Detections.from_ultralytics(results[0])

    kp.data["xyxy"] = det.xyxy

    return kp

def infer_model_type(model_path: Path) -> Literal["rf-detr", "yolo"] | None:
    if model_path.suffix == ".pth":
        return "rf-detr"
    elif model_path.suffix == ".pt":
        return "yolo"
    else:
        return None

def run(model_path: Path, image_paths: list[Path], input_space: Literal["canonical", "raw"], model_space: Literal["canonical", "raw"], model_type: Literal["rf-detr", "yolo"], out_dir: Path = Path("./results"), **kwargs):
    
    if model_space == "raw" and input_space == "canonical":
        print(f"Warning: unable to map 'canonical' input images to the requested 'raw' space for inference")

    match model_type:
        case "rf-detr":
            model = RFDETRKeypointPreview(pretrain_weights=model_path)
        case "yolo":
            model = YOLO(model_path)
        case _:
            raise SystemExit(f"Unknown model type '{model_type}'. Pass yolo or rf-detr explicitly to --model-type")

    json_dir = out_dir / model_path.stem / "vision-json"
    img_dir = out_dir /  model_path.stem / "annotated"

    json_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    should_rectify = input_space != "canonical" and model_space == "canonical"

    for image, path in load_images(image_paths, should_rectify):
        match model_type:
            case "rf-detr":
                keypoints = model.predict(image, **kwargs)
            case "yolo":
                keypoints = predict_yolo(model, image, **kwargs)
            case _:
                raise ValueError("model_type must be one of 'rf-detr', 'yolo'")
        
        if model_space != "canonical" and input_space != "canonical":
            rect = rectify(image)
            if rect is None or not rect.ok:
                print(f"Warning: image {path.stem} failed to rectify. Continuing...")
                continue
            canonical_keypoints = remap_keypoints(keypoints, rect.photo_to_canonical)
            canonical_image = rect.canonical
        else:
            canonical_keypoints = keypoints
            canonical_image = image

        components = keypoints_to_components(CLASS_NAMES, canonical_keypoints, pin_keypoint_masks=PIN_KEYPOINT_MASKS)

        h, w = canonical_image.shape[:2]

        annotated = draw_keypoints(image, keypoints)
        
        
        components_img = draw_components(components, res=np.array([w, h]))

        components_img_resized = resize_width(components_img, image.shape[1])

        vision_json_str = Netlist(
            components=components,
            name=f"{path.stem} resistors"
        ).to_json()

        out_json_file = json_dir / f"{path.stem}.json"
        out_json_file.write_text(vision_json_str)

        combined = np.vstack([image, annotated, components_img_resized])

        Image.fromarray(combined).save(img_dir / f"{path.stem}.png")

        print(f"Found {len(components)} components in {path}:")

        for c in components:
            print("-", c.type, [p.hole for p in c.pins.values()])

def main():
    parser = argparse.ArgumentParser(description="Detect resistors in an image using a pose model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="image files or directories")
    parser.add_argument("-o", "--output", type=Path, default=Path("results"), help="path to the top level output directory")
    parser.add_argument("--model", type=Path, default=Path("checkpoints/rf-detr-704-square-norm-synth10kv0-rp170.pth"), help="path to the rf-detr or yolo model checkpoint to use")
    parser.add_argument("--model-space", choices=["canonical", "raw"], default="canonical", help="what space is assumed by the model (rectified or not)")
    parser.add_argument("--input-space", choices=["canonical", "raw"], default="raw", help="what space the input images are in (rectified or not)")
    parser.add_argument("--model-type", choices=["yolo", "rf-detr"], help="autodetected from file extension if left empty")
    parser.add_argument("--conf", type=float, help="uses the default confidence from the checkpoint if left empty")

    args = parser.parse_args()

    image_paths = get_image_files(args.inputs)

    model_type = args.model_type if args.model_type is not None else infer_model_type(args.model)

    if model_type is None:
        raise SystemExit("Unable to determine model type from path. Pass yolo or rf-detr explicitly to --model-type")

    if not args.model.exists():
        raise SystemExit(f"Checkpoint path not found: {args.model}")

    extra_args = {}

    if args.conf is not None:
        match model_type:
            case "yolo":
                extra_args = {"conf": args.conf}
            case "rf-detr":
                extra_args = {"threshold": args.conf}

    run(args.model, image_paths, args.input_space, args.model_space, model_type=model_type, out_dir=args.output, **extra_args)

if __name__ == "__main__":
    main()



        

