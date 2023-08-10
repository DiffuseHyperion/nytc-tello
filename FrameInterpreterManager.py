import threading
import time
from collections import deque
from numpy import ndarray

from FrameInterpreter import FrameInterpreterPool, IncomingFrame
from VideoRead import VideoRead


class FrameInterpreterManager:
    def __init__(self, instances: int,
                 model_path: str,
                 label_path: str,
                 confidence_threshold: float,
                 video_read: VideoRead,
                 target_fps: int):

        self.pool = FrameInterpreterPool(instances, model_path, label_path, confidence_threshold)
        self.instances = instances
        self.video_read = video_read
        self.queue = deque()
        self.outgoing_queue = deque()
        self.should_stop = False
        self.target_fps = target_fps

        threading.Thread(target=self._manager).start()

    def _manager(self):
        # NEWEST --> OLDEST
        while not self.should_stop:
            print("Queueing frame!")
            incoming_frame = IncomingFrame(self.video_read.get_frame(), self.pool)
            self.queue.appendleft(incoming_frame)

            if len(self.queue) <= self.instances:
                incoming_frame.process_frame()

            outgoing_frame: IncomingFrame = self.queue[0]
            if outgoing_frame.is_done():
                # Show incoming frame
                new_image: IncomingFrame = self.queue.pop()
                new_frame: ndarray = new_image.get_processed_frame()
                self.outgoing_queue.appendleft(new_frame)

                # Start object detection on appropriate thread
                if len(self.queue) >= self.instances:
                    next_frame: IncomingFrame = self.queue[self.instances - 1]
                    next_frame.process_frame()
            time.sleep(1 / self.target_fps)

    def stop(self):
        self.should_stop = True

    def get_outgoing_queue(self):
        return self.outgoing_queue
