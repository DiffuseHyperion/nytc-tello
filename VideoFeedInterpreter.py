"""
Takes frames from video_feed and does tensorflow's magic tricks
Adapted from https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_stream.py
"""
import threading

#from tensorflow.lite.python.interpreter import Interpreter
from tflite_runtime.interpreter import Interpreter
import os
import numpy as np
import cv2

TF1_MODEL = 1
TF2_MODEL = 2


class VideoFeedIntepreter:
    def __init__(self, video_feed, model_dir, minimum_confidence, model_type):
        self.video_feed = video_feed
        self.frame = np.zeros([300, 400, 3], dtype=np.uint8)
        self.running = True
        self.minimum_confidence = minimum_confidence
        self.image_width = 960  # very real 720p video :))))
        self.image_height = 720

        path_to_model = os.path.join(model_dir, "model.tflite")
        path_to_labels = os.path.join(model_dir, "labels.txt")

        with open(path_to_labels, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if self.labels[0] == '???':
            del (self.labels[0])

        self.tfint = Interpreter(model_path=path_to_model)

        self.tfint.allocate_tensors()

        self.input_details = self.tfint.get_input_details()
        self.output_details = self.tfint.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

        self.input_mean = 127.5
        self.input_std = 127.5

        # Check output layer name to determine if this model was created with TF2 or TF1,
        # because outputs are ordered differently for TF2 and TF1 models
        self.outname = self.output_details[0]['name']

        if model_type == 2:
            # This is a TF2 model
            self.boxes_idx, self.classes_idx, self.scores_idx = 1, 3, 0
        elif model_type == 1:
            # This is a TF1 model
            self.boxes_idx, self.classes_idx, self.scores_idx = 0, 1, 2
        else:
            raise ValueError("Invalid model type given!")

        threading.Thread(target=self._frame_interpreter).start()

    def _frame_interpreter(self):
        while self.running:
            # Acquire frame and resize to expected shape [1xHxWx3]
            frame = self.video_feed.get_frame()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if self.floating_model:
                input_data = (np.float32(input_data) - self.input_mean) / self.input_std

            # Perform the actual detection by running the model with the image as input
            self.tfint.set_tensor(self.input_details[0]['index'], input_data)
            self.tfint.invoke()

            # Retrieve detection results
            boxes = self.tfint.get_tensor(self.output_details[self.boxes_idx]['index'])[
                0]  # Bounding box coordinates of detected objects
            classes = self.tfint.get_tensor(self.output_details[self.classes_idx]['index'])[
                0]  # Class index of detected objects
            scores = self.tfint.get_tensor(self.output_details[self.scores_idx]['index'])[
                0]  # Confidence of detected objects

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if (scores[i] > self.minimum_confidence) and (scores[i] <= 1.0):
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside
                    # of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1, (boxes[i][0] * self.image_height)))
                    xmin = int(max(1, (boxes[i][1] * self.image_width)))
                    ymax = int(min(self.image_height, (boxes[i][2] * self.image_height)))
                    xmax = int(min(self.image_width, (boxes[i][3] * self.image_width)))

                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                    # Draw label
                    object_name = self.labels[
                        int(classes[i])]  # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                    label_ymin = max(ymin, label_size[1] + 10)  # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                                  (xmin + label_size[0], label_ymin + base_line - 10), (255, 255, 255),
                                  cv2.FILLED)  # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                                2)  # Draw label text

            self.frame = frame

    def stop(self):
        self.running = False

    def get_frame(self):
        return self.frame
