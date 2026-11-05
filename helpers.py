from PIL import ImageFont
from breadboard.rectify import template_holes
from breadboard.netlist import Component, Pin

from collections import Counter

import numpy as np

import cv2
import copy

from PIL import Image, ImageDraw

import supervision as sv

def hole_at(x) -> str:
    """Inversion of ``template_holes()`` to find the closest hole name from a canonical pixel position"""
    holes = template_holes()
    mdist2 = np.inf
    closest = None
    for n, p in holes.items():
        dist2 = (p[0] - x[0])**2 + (p[1] - x[1])**2
        if dist2 < mdist2:
            mdist2 = dist2
            closest = n
    return closest

def transform_points_3x3(points, mat3):
    if points.size == 0:
        return points
    original_shape = points.shape
    points_transformed = points.reshape(-1, 1, 2)
    points_transformed = cv2.perspectiveTransform(points_transformed, mat3)
    return points_transformed.reshape(original_shape)

def resize_width(image: np.ndarray, target_width: int):
    h, w = image.shape[:2]
    factor = float(target_width) / float(w)
    return cv2.resize(image, (target_width, int(h * factor)), interpolation=cv2.INTER_NEAREST)

def remap_keypoints(keypoints: sv.KeyPoints, mat3: np.ndarray) -> sv.KeyPoints:
    """Apply a 3x3 transformation matrix to a supervision ``KeyPoints`` object.
    
    Args:
        keypoints: supervision ``KeyPoints`` object.
        mat3: 3x3 transformation matrix from source to target coordinates.

    Returns:
        A new ``KeyPoints`` object with transformed bounding box and keypoint positions. If present the original 
        bounding box is preserved after transformation as a quadrilateral in ``new_keypoints.data["xyxyxyxy"]``.

    Warning:
        ``keypoints.data["covariance"]`` is passed through as-is. Comparisons can be done in the raw image space, 
        or the conversion can be done separately.
    """

    new_keypoints = copy.deepcopy(keypoints)

    if "xyxy" in keypoints.data:
        x1, y1, x2, y2 = new_keypoints.data["xyxy"].T
        corners = np.stack([
            np.stack([x1, y1], axis=-1),   # top-left
            np.stack([x2, y1], axis=-1),   # top-right
            np.stack([x2, y2], axis=-1),   # bottom-right
            np.stack([x1, y2], axis=-1),   # bottom-left
        ], axis=1)                         # (n, 4, 2)
        warped = transform_points_3x3(corners, mat3)
        bmin = np.min(warped, axis=1) # (n, 2)
        bmax = np.max(warped, axis=1) # (n, 2)
        new_keypoints.data["xyxy"] = np.concatenate([bmin, bmax], axis=1)
        new_keypoints.data["xyxyxyxy"] = warped

    new_keypoints.xy = transform_points_3x3(new_keypoints.xy, mat3)

    return new_keypoints

def keypoints_to_components(class_names: list[str], keypoints: sv.KeyPoints, pin_keypoint_masks: np.ndarray | None = None, dedup=True) -> list[Component]:
    """Snap detected keypoints to breadboard pin positions and return a list of ``Breadboard-Classifier`` ``Component`` objects

    Args:
        class_names: globally exhaustive list that maps raw ``keypoints.class_id`` to a string name.
        keypoints: supervision KeyPoints in canonical pixel space.
            If originally detected in the raw image space they need to be first converted with ``remap_keypoints(keypoints, rect.photo_to_canonical)``.
        pin_keypoint_masks: (n_classes, n_keypoints) mask of which keypoints correspond to pin connections. If None all keypoints are assumed to be pins.
        dedup: merge detections that share the same set of pinholes.
    
    Returns:
        a list of Component objects extracted from keypoints. Pin confidence is the detection confidence for the whole component.
    """

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
    det.xyxy = kp.data.get("xyxy", det.xyxy)
    return det

def draw_keypoints(image_rgb: np.ndarray, keypoints: sv.KeyPoints) -> np.ndarray:
    """Draws detections from a supervision ``KeyPoints`` object onto a copy of the source image using supervision annotators"""
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
    
def draw_components(components: list[Component], res=np.array([1024, 340])) -> np.ndarray:
    """Draws components from a list of ``Breadboard-Classifier`` ``Component`` objects
    
    Args:
        components: a list of ``Breadboard-Classifier`` ``Component`` objects
        res: the canonical resolution used by the classifier and ``template_holes()``
    
    Returns:
        The annotated image
    """
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
