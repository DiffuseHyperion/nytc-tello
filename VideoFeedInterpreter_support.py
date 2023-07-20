"""
Takes frames from video_feed and does tensorflow's magic tricks
Adapted from https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_stream.py
"""
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

import os
import numpy as np
import cv2
import threading

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red


class VideoFeedInterpreter:
    def __init__(self, video_feed, model_dir, minimum_confidence):
        self.video_feed = video_feed
        self.frame = np.zeros([300, 400, 3], dtype=np.uint8)
        self.running = True
        self.image_width = 960  # very real 720p video :))))
        self.image_height = 720

        path_to_model = os.path.join(model_dir, "model.tflite")
        path_to_labels = os.path.join(model_dir, "labels.txt")

        base_options = core.BaseOptions(
            file_name=path_to_model, use_coral=False, num_threads=1)
        detection_options = processor.DetectionOptions(
            max_results=1, score_threshold=minimum_confidence)
        options = vision.ObjectDetectorOptions(
            base_options=base_options, detection_options=detection_options)

        self.detector = vision.ObjectDetector.create_from_options(options)

        threading.Thread(target=self._frame_interpreter).start()

    def _frame_interpreter(self):
        while self.running:
            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = self.video_feed.get_frame()

            # Convert the image from BGR to RGB as required by the TFLite model.
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create TensorImage from the RGB image
            tensor_image = vision.TensorImage.create_from_array(rgb_image)
            # List classification results
            detection_result = self.detector.detect(tensor_image)

            for detection in detection_result.detections:
                # Draw bounding_box
                bbox = detection.bounding_box
                start_point = bbox.origin_x, bbox.origin_y
                end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
                cv2.rectangle(frame, start_point, end_point, _TEXT_COLOR, 3)

                # Draw label and score
                category = detection.categories[0]
                category_name = category.category_name
                probability = round(category.score, 2)
                result_text = category_name + ' (' + str(probability) + ')'
                text_location = (_MARGIN + bbox.origin_x,
                                 _MARGIN + _ROW_SIZE + bbox.origin_y)
                cv2.putText(frame, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                            _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)

            self.frame = frame

    def stop(self):
        self.running = False

    def get_frame(self):
        return self.frame
