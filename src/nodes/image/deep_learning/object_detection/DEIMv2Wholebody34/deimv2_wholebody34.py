#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from dataclasses import dataclass
from typing import List, Optional, Dict

import cv2
import numpy as np
import onnxruntime


@dataclass(frozen=False)
class Box:
    classid: int
    score: float
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int
    generation: int = -1  # -1: Unknown, 0: Adult, 1: Child
    gender: int = -1  # -1: Unknown, 0: Male, 1: Female
    handedness: int = -1  # -1: Unknown, 0: Left, 1: Right
    head_pose: int = -1  # -1: Unknown, 0: Front, 1: Right-Front, 2: Right-Side, 3: Right-Back, 4: Back, 5: Left-Back, 6: Left-Side, 7: Left-Front
    is_used: bool = False
    person_id: int = -1


# Edge connections for skeleton
EDGES = [
    (21, 22), (21, 22),  # collarbone -> shoulder (left and right)
    (21, 23),            # collarbone -> solar_plexus
    (22, 24), (22, 24),  # shoulder -> elbow (left and right)
    (22, 30), (22, 30),  # shoulder -> hip_joint (left and right)
    (24, 25), (24, 25),  # elbow -> wrist (left and right)
    (23, 29),            # solar_plexus -> abdomen
    (29, 30), (29, 30),  # abdomen -> hip_joint (left and right)
    (30, 31), (30, 31),  # hip_joint -> knee (left and right)
    (31, 32), (31, 32),  # knee -> ankle (left and right)
]

# Head pose colors
BOX_COLORS = [
    [(216, 67, 21), "Front"],
    [(255, 87, 34), "Right-Front"],
    [(123, 31, 162), "Right-Side"],
    [(255, 193, 7), "Right-Back"],
    [(76, 175, 80), "Back"],
    [(33, 150, 243), "Left-Back"],
    [(156, 39, 176), "Left-Side"],
    [(0, 188, 212), "Left-Front"],
]


class DEIMv2Wholebody34:
    def __init__(
        self,
        model_path: str = 'deimv2_hgnetv2_n_wholebody34_680query_n_batch_640x640.onnx',
        obj_class_score_th: float = 0.35,
        attr_class_score_th: float = 0.70,
        keypoint_th: float = 0.35,
        providers: Optional[List] = None,
    ):
        if providers is None:
            providers = [
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]

        self._obj_class_score_th = obj_class_score_th
        self._attr_class_score_th = attr_class_score_th
        self._keypoint_th = keypoint_th

        # Load ONNX model
        session_option = onnxruntime.SessionOptions()
        session_option.log_severity_level = 3
        onnxruntime.set_default_logger_severity(3)

        self._interpreter = onnxruntime.InferenceSession(
            model_path,
            sess_options=session_option,
            providers=providers,
        )

        self._input_names = [
            inp.name for inp in self._interpreter.get_inputs()
        ]
        self._output_names = [
            out.name for out in self._interpreter.get_outputs()
        ]

        # Get input shape (NCHW format)
        # Some models have dynamic input shape like ['N', 'C', 'H', 'W']
        # In that case, extract size from model filename (e.g., 320x320, 416x416, 640x640)
        input_shape = self._interpreter.get_inputs()[0].shape
        try:
            self._input_height = int(input_shape[2])
            self._input_width = int(input_shape[3])
        except (ValueError, TypeError):
            # Extract from filename like "..._320x320.onnx" or "..._416x416_nopre.onnx"
            import re
            match = re.search(r'(\d+)x(\d+)', model_path)
            if match:
                self._input_height = int(match.group(1))
                self._input_width = int(match.group(2))
            else:
                # Default to 640x640
                self._input_height = 640
                self._input_width = 640

        # Get input dtype
        onnx_dtypes_to_np_dtypes = {
            "tensor(float)": np.float32,
            "tensor(uint8)": np.uint8,
            "tensor(int8)": np.int8,
        }
        self._input_dtypes = [
            onnx_dtypes_to_np_dtypes[inp.type] for inp in self._interpreter.get_inputs()
        ]

    def __call__(
        self,
        image: np.ndarray,
        disable_generation_identification_mode: bool = False,
        disable_gender_identification_mode: bool = False,
        disable_left_and_right_hand_identification_mode: bool = False,
        disable_headpose_identification_mode: bool = False,
    ) -> List[Box]:
        temp_image = copy.deepcopy(image)

        # PreProcess
        resized_image = self._preprocess(temp_image)

        # Inference
        inference_image = np.asarray([resized_image], dtype=self._input_dtypes[0])
        outputs = self._interpreter.run(
            self._output_names,
            {self._input_names[0]: inference_image},
        )
        boxes = outputs[0][0]

        # PostProcess
        result_boxes = self._postprocess(
            image=temp_image,
            boxes=boxes,
            disable_generation_identification_mode=disable_generation_identification_mode,
            disable_gender_identification_mode=disable_gender_identification_mode,
            disable_left_and_right_hand_identification_mode=disable_left_and_right_hand_identification_mode,
            disable_headpose_identification_mode=disable_headpose_identification_mode,
        )

        return result_boxes

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        # Resize to model input size
        resized_image = cv2.resize(
            image,
            (self._input_width, self._input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        # HWC -> CHW
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = np.ascontiguousarray(resized_image, dtype=np.float32)
        return resized_image

    def _postprocess(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        disable_generation_identification_mode: bool,
        disable_gender_identification_mode: bool,
        disable_left_and_right_hand_identification_mode: bool,
        disable_headpose_identification_mode: bool,
    ) -> List[Box]:
        image_height = image.shape[0]
        image_width = image.shape[1]

        result_boxes: List[Box] = []

        box_score_threshold = min([
            self._obj_class_score_th,
            self._attr_class_score_th,
            self._keypoint_th
        ])

        if len(boxes) > 0:
            scores = boxes[:, 5:6]
            keep_idxs = scores[:, 0] > box_score_threshold
            boxes_keep = boxes[keep_idxs, :]

            if len(boxes_keep) > 0:
                for box in boxes_keep:
                    classid = int(box[0])
                    x_min = int(max(0, box[1]) * image_width)
                    y_min = int(max(0, box[2]) * image_height)
                    x_max = int(min(box[3], 1.0) * image_width)
                    y_max = int(min(box[4], 1.0) * image_height)
                    cx = (x_min + x_max) // 2
                    cy = (y_min + y_max) // 2
                    score = float(box[5])
                    result_boxes.append(
                        Box(
                            classid=classid,
                            score=score,
                            x1=x_min,
                            y1=y_min,
                            x2=x_max,
                            y2=y_max,
                            cx=cx,
                            cy=cy,
                        )
                    )

                # Object filter
                result_boxes = [
                    box for box in result_boxes
                    if (box.classid in [0, 5, 6, 7, 16, 17, 18, 19, 20, 26, 27, 28, 33]
                        and box.score >= self._obj_class_score_th)
                    or box.classid not in [0, 5, 6, 7, 16, 17, 18, 19, 20, 26, 27, 28, 33]
                ]
                # Attribute filter
                result_boxes = [
                    box for box in result_boxes
                    if (box.classid in [1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15]
                        and box.score >= self._attr_class_score_th)
                    or box.classid not in [1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15]
                ]
                # Keypoint filter
                result_boxes = [
                    box for box in result_boxes
                    if (box.classid in [21, 22, 23, 24, 25, 29, 30, 31, 32]
                        and box.score >= self._keypoint_th)
                    or box.classid not in [21, 22, 23, 24, 25, 29, 30, 31, 32]
                ]

                # Generation merge (Adult/Child)
                if not disable_generation_identification_mode:
                    body_boxes = [box for box in result_boxes if box.classid == 0]
                    generation_boxes = [box for box in result_boxes if box.classid in [1, 2]]
                    self._find_most_relevant_obj(base_objs=body_boxes, target_objs=generation_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [1, 2]]

                # Gender merge (Male/Female)
                if not disable_gender_identification_mode:
                    body_boxes = [box for box in result_boxes if box.classid == 0]
                    gender_boxes = [box for box in result_boxes if box.classid in [3, 4]]
                    self._find_most_relevant_obj(base_objs=body_boxes, target_objs=gender_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [3, 4]]

                # HeadPose merge
                if not disable_headpose_identification_mode:
                    head_boxes = [box for box in result_boxes if box.classid == 7]
                    headpose_boxes = [box for box in result_boxes if box.classid in [8, 9, 10, 11, 12, 13, 14, 15]]
                    self._find_most_relevant_obj(base_objs=head_boxes, target_objs=headpose_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [8, 9, 10, 11, 12, 13, 14, 15]]

                # Left/Right hand merge
                if not disable_left_and_right_hand_identification_mode:
                    hand_boxes = [box for box in result_boxes if box.classid == 26]
                    left_right_hand_boxes = [box for box in result_boxes if box.classid in [27, 28]]
                    self._find_most_relevant_obj(base_objs=hand_boxes, target_objs=left_right_hand_boxes)
                result_boxes = [box for box in result_boxes if box.classid not in [27, 28]]

                # Keypoints NMS
                for target_classid in [21, 22, 23, 24, 25, 29, 30, 31, 32]:
                    keypoints_boxes = [box for box in result_boxes if box.classid == target_classid]
                    filtered_keypoints_boxes = self._nms(target_objs=keypoints_boxes, iou_threshold=0.20)
                    result_boxes = [box for box in result_boxes if box.classid != target_classid]
                    result_boxes = result_boxes + filtered_keypoints_boxes

        return result_boxes

    def _find_most_relevant_obj(
        self,
        base_objs: List[Box],
        target_objs: List[Box],
    ):
        for base_obj in base_objs:
            most_relevant_obj: Box = None
            best_score = 0.0
            best_iou = 0.0
            best_distance = float('inf')

            for target_obj in target_objs:
                distance = ((base_obj.cx - target_obj.cx)**2 + (base_obj.cy - target_obj.cy)**2)**0.5
                if not target_obj.is_used and distance <= 10.0:
                    if target_obj.score >= best_score:
                        iou = self._calculate_iou(base_obj=base_obj, target_obj=target_obj)
                        if iou > best_iou:
                            most_relevant_obj = target_obj
                            best_iou = iou
                            best_distance = distance
                            best_score = target_obj.score
                        elif iou > 0.0 and iou == best_iou:
                            if distance < best_distance:
                                most_relevant_obj = target_obj
                                best_distance = distance
                                best_score = target_obj.score

            if most_relevant_obj:
                if most_relevant_obj.classid == 1:
                    base_obj.generation = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 2:
                    base_obj.generation = 1
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 3:
                    base_obj.gender = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 4:
                    base_obj.gender = 1
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 8:
                    base_obj.head_pose = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 9:
                    base_obj.head_pose = 1
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 10:
                    base_obj.head_pose = 2
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 11:
                    base_obj.head_pose = 3
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 12:
                    base_obj.head_pose = 4
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 13:
                    base_obj.head_pose = 5
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 14:
                    base_obj.head_pose = 6
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 15:
                    base_obj.head_pose = 7
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 27:
                    base_obj.handedness = 0
                    most_relevant_obj.is_used = True
                elif most_relevant_obj.classid == 28:
                    base_obj.handedness = 1
                    most_relevant_obj.is_used = True

    def _nms(
        self,
        target_objs: List[Box],
        iou_threshold: float,
    ) -> List[Box]:
        filtered_objs: List[Box] = []
        sorted_objs = sorted(target_objs, key=lambda box: box.score, reverse=True)

        while sorted_objs:
            current_box = sorted_objs.pop(0)
            if current_box.is_used:
                continue

            filtered_objs.append(current_box)
            current_box.is_used = True

            remaining_boxes = []
            for box in sorted_objs:
                if not box.is_used:
                    iou_value = self._calculate_iou(base_obj=current_box, target_obj=box)
                    if iou_value >= iou_threshold:
                        box.is_used = True
                    else:
                        remaining_boxes.append(box)
            sorted_objs = remaining_boxes

        return filtered_objs

    def _calculate_iou(
        self,
        base_obj: Box,
        target_obj: Box,
    ) -> float:
        inter_xmin = max(base_obj.x1, target_obj.x1)
        inter_ymin = max(base_obj.y1, target_obj.y1)
        inter_xmax = min(base_obj.x2, target_obj.x2)
        inter_ymax = min(base_obj.y2, target_obj.y2)

        if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        area1 = (base_obj.x2 - base_obj.x1) * (base_obj.y2 - base_obj.y1)
        area2 = (target_obj.x2 - target_obj.x1) * (target_obj.y2 - target_obj.y1)

        iou = inter_area / float(area1 + area2 - inter_area)
        return iou
