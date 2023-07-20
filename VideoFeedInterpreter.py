import os
import threading
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter


class VideoFeedInterpreter:

    def __init__(self, video_feed, model_dir, minimum_confidence):
        self.video_feed = video_feed
        self.frame = np.zeros([300, 400, 3], dtype=np.uint8)
        self.running = True
        self.minimum_confidence = minimum_confidence
        self.image_width = 960  # very real 720p video :))))
        self.image_height = 720
        self.input_mean = 127.5
        self.input_std = 127.5

        path_to_model = os.path.join(model_dir, "model.tflite")
        path_to_labels = os.path.join(model_dir, "labels.txt")

        with open(path_to_labels, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        self.interpreter = Interpreter(model_path=path_to_model)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # check the type of the input tensor
        self.floating_model = self.input_details[0]['dtype'] == np.float32

        # NxHxWxC, H:1, W:2
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]

        threading.Thread(target=self._frame_interpreter_thread).start()

    def _frame_interpreter_thread(self):
        frame = self.video_feed.get_frame()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        # Perform the actual detection by running the model with the image as input
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        print(output_data)
        results = np.squeeze(output_data)
        print(results)
        top_k = results.argsort()[-5:][::-1]

        for i in top_k:
            if self.floating_model:
                print('{:08.6f}: {}'.format(float(results[i]), self.labels[i]))
            else:
                print('{:08.6f}: {}'.format(float(results[i] / 255.0), self.labels[i]))
