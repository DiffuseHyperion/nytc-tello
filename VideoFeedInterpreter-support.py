"""
Takes frames from video_feed and does tensorflow's magic tricks
Adapted from https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_stream.py
"""
import threading

#from tensorflow.lite.python.interpreter import Interpreter
#from tflite_runtime.interpreter import Interpreter
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

import os
import numpy as np
import cv2


class VideoFeedIntepreter:
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

        # Enable Coral by this setting
        classification_options = processor.ClassificationOptions(
            max_results=1, score_threshold=minimum_confidence)
        options = vision.ImageClassifierOptions(
            base_options=base_options, classification_options=classification_options)

        self.classifier = vision.ImageClassifier.create_from_options(options)

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
            categories = self.classifier.classify(tensor_image)
            for idx, category in enumerate(categories.classifications[0].categories):
                category_name = category.category_name
                score = round(category.score, 2)
                result_text = category_name + ' (' + str(score) + ')'
                text_location = (24, (idx + 2) * 20)
                cv2.putText(frame, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 0, 255), 1)

            self.frame = frame

    def stop(self):
        self.running = False

    def get_frame(self):
        return self.frame
