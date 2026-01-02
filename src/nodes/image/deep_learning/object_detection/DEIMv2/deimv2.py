#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime


class DEIMv2:
    def __init__(
        self,
        model_path: str = 'deimv2_hgnetv2_n_coco.onnx',
        class_score_th: float = 0.35,
        providers: Optional[List] = None,
    ):
        if providers is None:
            providers = [
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]

        self._class_score_th = class_score_th

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
        input_shape = self._interpreter.get_inputs()[0].shape
        try:
            self._input_height = int(input_shape[2])
            self._input_width = int(input_shape[3])
        except (ValueError, TypeError):
            # Extract from filename like "..._640x640.onnx"
            match = re.search(r'(\d+)x(\d+)', model_path)
            if match:
                self._input_height = int(match.group(1))
                self._input_width = int(match.group(2))
            else:
                # Default to 640x640
                self._input_height = 640
                self._input_width = 640

    def __call__(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        image: np.ndarray
            Input BGR image

        Returns
        -------
        bboxes: np.ndarray
            Bounding boxes [N, 4] (x1, y1, x2, y2)
        scores: np.ndarray
            Confidence scores [N]
        class_ids: np.ndarray
            Class IDs [N]
        """
        image_height, image_width = image.shape[0], image.shape[1]

        # PreProcess
        input_image = self._preprocess(image)

        # Original size for output scaling
        orig_target_sizes = np.array([[image_width, image_height]], dtype=np.int64)

        # Inference
        outputs = self._interpreter.run(
            None,
            {
                "images": input_image,
                "orig_target_sizes": orig_target_sizes,
            },
        )
        labels, bboxes_out, scores_out = outputs

        # PostProcess
        bboxes, scores, class_ids = self._postprocess(
            labels=labels[0],
            bboxes=bboxes_out[0],
            scores=scores_out[0],
            image_width=image_width,
            image_height=image_height,
        )

        return bboxes, scores, class_ids

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        # BGR -> RGB
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize to model input size
        input_image = cv2.resize(
            input_image,
            (self._input_width, self._input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        # Normalize to 0-1
        input_image = input_image.astype(np.float32) / 255.0
        # HWC -> CHW
        input_image = input_image.transpose((2, 0, 1))
        # Add batch dimension
        input_image = np.expand_dims(input_image, axis=0)
        return input_image

    def _postprocess(
        self,
        labels: np.ndarray,
        bboxes: np.ndarray,
        scores: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        labels: np.ndarray
            Class labels [N]
        bboxes: np.ndarray
            Bounding boxes [N, 4] (x1, y1, x2, y2) in pixel coordinates
        scores: np.ndarray
            Confidence scores [N]

        Returns
        -------
        bboxes: np.ndarray
        scores: np.ndarray
        class_ids: np.ndarray
        """
        result_bboxes = []
        result_scores = []
        result_class_ids = []

        for label, bbox, score in zip(labels, bboxes, scores):
            if score < self._class_score_th:
                continue

            x1 = int(max(0, bbox[0]))
            y1 = int(max(0, bbox[1]))
            x2 = int(min(bbox[2], image_width))
            y2 = int(min(bbox[3], image_height))

            result_bboxes.append([x1, y1, x2, y2])
            result_scores.append(float(score))
            result_class_ids.append(int(label))

        return np.array(result_bboxes), np.array(result_scores), np.array(result_class_ids)
