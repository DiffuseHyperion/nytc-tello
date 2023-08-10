import threading
from asyncio import Future
from sys import platform
import numpy as np
import cv2
from collections import deque

if platform == "linux" or platform == "linux2":
    from tflite_runtime.interpreter import Interpreter
elif platform == "win32":
    from tensorflow.lite.python.interpreter import Interpreter


def init_interpreter(model_path: str):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details, output_details = interpreter.get_input_details(), interpreter.get_output_details()
    return interpreter, input_details, output_details


class FrameInterpreter:
    def __init__(self, model_path: str, label_path: str, confidence_threshold: float):
        self.interpreter, self.input_details, self.output_details = init_interpreter(model_path)
        self.confidence_threshold = confidence_threshold
        with open(label_path, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]
            f.close()

    def process_frame(self, original_frame: np.ndarray):
        future = Future()
        threading.Thread(target=self._process_frame_thread, args=[original_frame, future]).start()

        return future

    def _process_frame_thread(self, original_frame: np.ndarray, future: Future):
        original_shape = np.shape(original_frame)
        input_shape = self.input_details[0]['shape']
        new_image = cv2.resize(original_frame, (input_shape[1], input_shape[2]))

        self.interpreter.set_tensor(self.input_details[0]['index'], [new_image])
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.output_details[0]['index']).squeeze()
        classes = self.interpreter.get_tensor(self.output_details[1]['index']).squeeze()
        scores = self.interpreter.get_tensor(self.output_details[2]['index']).squeeze()

        for i in range(len(scores)):
            if scores[i] > self.confidence_threshold:
                # Unnormalize boundaries
                unnormed_coords = boxes[i] * input_shape[1]
                start_point = (int(unnormed_coords[1]), int(unnormed_coords[0]))
                end_point = (int(unnormed_coords[3]), int(unnormed_coords[2]))
                # Draw bounding box
                drawn = cv2.rectangle(new_image, start_point, end_point, color=(0, 255, 0), thickness=2)
                # Add label and score
                img_text = f"{self.labels[int(classes[i])]}: {scores[i]:.3f}"
                output_label = cv2.putText(new_image, img_text, (int(unnormed_coords[1]), int(unnormed_coords[0]) + 15),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                break

        new_image = cv2.resize(new_image, (original_shape[1], original_shape[0]))
        new_image = np.rot90(new_image)
        new_image = np.flipud(new_image)

        future.set_result(new_image)


class FrameInterpreterPool:
    def __init__(self,
                 instances: int,
                 model_path: str,
                 label_path: str,
                 confidence_threshold: float):
        self.pool = []
        self.busy_interpreters = 0
        self.max_busy_interpreters = instances
        for i in range(instances):
            self.pool.append(FrameInterpreter(model_path, label_path, confidence_threshold))

    def run(self, original_frame: np.ndarray):
        if self.busy_interpreters >= self.max_busy_interpreters:
            raise Exception("All interpreters were used up!")
        else:
            interpreter: FrameInterpreter = self.pool[self.busy_interpreters]
            self.busy_interpreters += 1
            future: Future = interpreter.process_frame(original_frame)

            def _done_callback(res: Future):
                self.busy_interpreters -= 1

            future.add_done_callback(_done_callback)
            return future

    def get_used_interpreters(self):
        return self.busy_interpreters


class IncomingFrame:
    def __init__(self, original_frame: np.ndarray, pool: FrameInterpreterPool):
        self.interpreter = None
        self.original_frame = original_frame
        self.processed_frame = None
        self.done = False
        self.pool = pool

    def process_frame(self):
        future = self.pool.run(self.original_frame)

        def _process_callback(res: Future):
            self.processed_frame = res.result()
            self.done = True

        future.add_done_callback(_process_callback)

    def get_processed_frame(self):
        if self.processed_frame is None:
            raise Exception("This frame has not finished processing!")
        return self.processed_frame

    def is_done(self):
        return self.done
