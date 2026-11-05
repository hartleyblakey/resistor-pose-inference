from typing import Literal
from PIL import ImageFont
from breadboard.rectify import template_holes
from breadboard.netlist import Component, Netlist, Pin

from collections import Counter

import numpy as np

from PIL import Image, ImageDraw

import supervision as sv

def hole_at(x) -> str:
    holes = template_holes()
    mdist2 = np.inf
    closest = None
    for n, p in holes.items():
        dist2 = (p[0] - x[0])**2 + (p[1] - x[1])**2
        if dist2 < mdist2:
            mdist2 = dist2
            closest = n
    return closest

def keypoints_to_components(class_names: list[str], keypoints: sv.KeyPoints, pin_keypoint_masks: np.ndarray | None = None, dedup=True) -> list[Component]:

    ids = []
    all_holes = []
    best_detections: dict[str, tuple[int, float]] = {}
    collisions = Counter()

    for i, item in enumerate(keypoints):
        confidence=float(keypoints.detection_confidence[i])
        cid = keypoints.class_id[i]
        xy = keypoints.xy[i]
        
        if pin_keypoint_masks is not None:
            pin_mask = pin_keypoint_masks[cid]
        else:
            pin_mask = np.ones(len(xy), dtype=bool)

        holes = [
            hole_at(pos) for pos in xy[pin_mask]
        ]

        all_holes.append(holes)

        span = "-".join(sorted(holes))
        id_str = class_names[cid] + "-" + span

        collisions[id_str] += 1

        if not dedup:
            id_str += "-" + str(collisions[id_str])

        if id_str not in best_detections or confidence > best_detections[id_str][1]:
            best_detections[id_str] = (i, confidence)
        
        ids.append(id_str)

    components = []

    for i, _ in best_detections.values():
        confidence=float(keypoints.detection_confidence[i])

        component = Component(
            id=ids[i],
            type=class_names[keypoints.class_id[i]],

            # keypoints.data["covariance"] might be useful for a per-keypoint score later
            # this just reuses the overall detection confidence. All keypoints of an instance have the same value
            pins={
                str(j + 1): Pin(hole=h, confidence=confidence) for j, h in enumerate(all_holes[i])
            }
        )

        components.append(component)

    return components

# like KeyPoints.as_detections() but passes through the detected bounding box
def keypoints_to_detections(kp: sv.KeyPoints) -> sv.Detections:
    det = kp.as_detections()
    det.xyxy = kp.data["xyxy"]
    return det

def draw_keypoints(image_rgb: np.ndarray, keypoints: sv.KeyPoints) -> np.ndarray:
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    vertex_annotator = sv.VertexAnnotator(radius=2, color=sv.Color.ROBOFLOW)

    ellipse_annotator = sv.VertexEllipseAreaAnnotator(opacity=0.4)
    
    detections = keypoints_to_detections(keypoints)

    annotated = image_rgb.copy()

    if "covariance" in keypoints.data:
        annotated = ellipse_annotator.annotate(annotated, keypoints)

    annotated = box_annotator.annotate(annotated, detections)
    if detections.confidence is not None:
        annotated = label_annotator.annotate(
            annotated, detections,
            labels=[f"{c:.2f}" for c in detections.confidence],
        )
    
    annotated = vertex_annotator.annotate(annotated, keypoints)

    return annotated
    
def draw_components(components: list[Component], res=np.array([1280, 340])):

    arr = np.ones((res[1], res[0], 3), dtype=np.uint8) * 128
    image = Image.fromarray(arr)

    draw = ImageDraw.Draw(image)

    holes = template_holes()

    font = ImageFont.load_default(size=16)

    for xy in holes.values():
        draw.circle(xy, 3, fill=(16, 16, 16))
    
    for c in components:
        line_seg = []
        for pn, p in c.pins.items():
            xy = holes[p.hole]
            draw.circle(xy, 5, fill=(64, 255, 64))
            draw.text(xy, p.hole, font=font)
            line_seg.append(xy)

        draw.line(line_seg, fill=(255, 64, 64))

    return np.array(image)
