# Pose Model Inference Example

This repo contains helpers to work with YOLO, supervision KeyPoints, and Breadboard-Classifier types. It also contains a demo CLI that takes in paths to images or directories of images and writes debug images annotated with their detections, alongside a minimal vision model json from the Breadboard-Classifier repo.

It basically just snaps the detected keypoints to their nearest neighbor pinhole position


## Setup
clone the repo and move the checkpoints you want from https://drive.google.com/drive/folders/1q1aeZAD8EObkU-Xv94Z3PWu0Ct1-8Mno?usp=drive_link into `resistor-pose-inference/checkpoints`

Then:
```bash
cd resistor-pose-inference
uv sync
```
Will install the dependencies. This may take a while if you have slow internet and don't happen to already have the correct torch and cuda versions installed


## Usage

```
uv run inference.py examples --model=checkpoints/rf-detr-704-square-norm-synth10kv0-rp170.pth
```

will write annotated images and minimal vision model json files to subdirectories of `./results` by default. The output directory can be changed with `-o path/to/dir`

The json files can be loaded by `breadboard.netlist.Netlist.from_json`


## Options

```
usage: inference.py [-h] [-o OUTPUT] [--model MODEL] [--model-space {canonical,raw}] [--input-space {canonical,raw}] [--model-type {yolo,rf-detr}] [--conf CONF] inputs [inputs ...]

Detect resistors in an image using a pose model

positional arguments:
  inputs                image files or directories

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   path to the top level output directory (default: results)
  --model MODEL         path to the rf-detr or yolo model checkpoint to use (default: checkpoints/rf-detr-704-square-norm-synth10kv0-rp170.pth)
  --model-space {canonical,raw}
                        what space is assumed by the model (rectified or not) (default: canonical)
  --input-space {canonical,raw}
                        what space the input images are in (rectified or not) (default: raw)
  --model-type {yolo,rf-detr}
                        autodetected from file extension if left empty (default: None)
  --conf CONF           uses the default confidence from the checkpoint if left empty (default: None)
```

## API Example

If you don't care about paving over the differences between checkpoints this is a smaller example than `inference.py` for running the model that shows the main steps.

```python
from breadboard.rectify import load_photo, rectify
from rfdetr import RFDETRKeypointPreview
from helpers import keypoints_to_components

model = RFDETRKeypointPreview(pretrain_weights="checkpoints/rf-detr-704-square-norm-synth10kv0-rp170.pth")

class_names = ["resistor"]

rect = rectify(load_photo("examples/_MG_8272.JPG"))
if rect is None or not rect.ok:
    raise SystemExit("Failed to rectify the image")

rectified_image = rect.canonical

keypoints = model.predict(rectified_image)

for c in keypoints_to_components(class_names, keypoints):
    print(c.type, [p.hole for p in c.pins.values()])

# resistor ['p1+:9', 'e3']
# resistor ['p1+:45', 'g42']
# resistor ['d28', 'c23']
# resistor ['h45', 'h34']
# resistor ['e16', 'i16']
```

Basically just rectify the image before running it through the model if that's what it was trained on. `keypoints_to_components` is a simple helper that converts `supervision` `KeyPoints` objects to lists of `Component`s from Breadboard-Classifier. In the process it also snaps the raw predicted keypoint positions to their closest breadboard pinhole, and merges detections that share identical pinholes.

## Checkpoints

| Checkpoint                                   | --model-space | training       | F1*     | res       | notes                                                               |
|----------------------------------------------|---------------|----------------|---------|-----------|---------------------------------------------------------------------|
| rf-detr-704-square-norm-rp170.pth            | canonical     | real only      | 0.784   | 704x704   |                                                                     |
| rf-detr-704-square-norm-synth10kv0-rp170.pth | canonical     | synthetic+real | 0.913   | 704x704   |                                                                     |
| rf-detr-704-square-norm-synth10kv0.pth       | raw           | synthetic only | 0.760** | 704x704   |                                                                     |
| yolo26s_aug_closs_v1-3_rp170.pt              | canonical     | real only      | 0.665   | 1280x1280 | older checkpoint, not sure exactly how it was configured or trained |
| yolo26s_aug_photo_flip_synth10kv0-rp170-2.pt | canonical     | synthetic+real | 0.830   | 1280x1280 | retrained for more epochs with a lower learning rate                |
| yolo26s_aug_photo_flip_synth10kv0-rp170.pt   | canonical     | synthetic+real | 0.843   | 1280x1280 |                                                                     |
| yolo26s_aug_photo_flip_synth10kv0.pt         | raw           | synthetic only | 0.833** | 1280x1280 |                                                                     |

\* Take the F1 scores with a massive grain of salt because of the tiny val set from the same cameras, components, and (mostly) environments as the training data. Detections were deduplicated by identical hole coverage and counted as correct if both holes of the detected resistor matched a label.

\*\* The synthetic models were trained on raw images but evaluated on rectified images like the rest of the models. This might deflate their scores somewhat because they had never seen images so zoomed in.

Other Notes for all checkpoints:
- I think the underlying ultralytics/roboflow inference code handles resizing and preprocessing automatically so you shouldn't need to care about the resolution of the images you feed it
- canonical is the default for --model-space if you don't pass anything
- "raw" images are more like a close crop since I tried to take well centered easy images where the breadboard was large in the frame

I don't really have much to say about the checkpoint training because it was so poorly planned and documented. Probably just that it's much better to spend time taking more pictures than tweaking augmentation configs aimlessly by hand. They used heavy photometric augmentation and only flip + the default geometric augmentations for YOLO.

Because resistors are symmetric the choice of which keypoint is kp0 is arbitrary. For all these checkpoints, the datasets were preprocessed to sort the keypoints by x position, so for each resistor kp0 is the keypoint with the smallest x coordinate and kp1 is the keypoint with the largest. Since resistors are most likely going to be axis-aligned with the board, I tried training models sorted by x+y but ran into problems with the built in augmentations not preserving that identity through nonuniform scale and flips. It might be a good idea to either stay simple or lean more on AI to do this kind of troubleshooting since it can take the time to read through the underlying training and augmentation code.

## Limitations

### Checkpoint Limitations:
1. The model struggles with vertical resistors. It can't predict which end is kp0 and which is kp1, they are sorted by x coordinate
2. Low accuracy in general
3. Poorly evaluated and undocumented training
4. Not trained on any real kit photos or components
            
    Performance is evaluated on the random collection of resistors and jumper wires I had lying around and not on the kit components

5. Only trained on resistors
6. Tiny dataset of 170 breadboards, with the validation set containing only ~35

### Code Limitations:
1. No per-keypoint confidence or fault detection of any kind

    It might be important to look into downweighting detections that overlap some of their keypoints with other, higher confidence detections

    It might be possible to extract a useful confidence signal from keypoint snapping distance and the covariance matrix. The latter is not reliable with these checkpoints, but that might change with a proper dataset

2. Very basic deduplication that removes detections that share all pinholes only

3. Largely assumes classes will be resistors or resistor-like


