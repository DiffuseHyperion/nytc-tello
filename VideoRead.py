"""
A custom implementation of djitellopy's BackgroundFrameRead.
Instead of throwing an error lacking frames to decode, it will instead skip decoding for the current loop.
Get current frame with video_feed.frame
"""

import threading
import numpy as np
import av
import time
from djitellopy import TelloException


class VideoRead:
    def __init__(self, tello_object):
        self.running = True
        self.tello = tello_object
        # np.zeros note: tuple goes like this: (amount of arrays, height of array, width of array)
        self.frame = np.zeros([300, 400, 3], dtype=np.uint8)

        self.tello.streamoff()
        self.tello.streamon()

        time.sleep(1)

        tries = 2
        while True:
            try:
                self.container = av.open(self.tello.get_udp_video_address(), timeout=(5, None))
            except av.error.ExitError:
                tries -= 1
                if tries <= 0:
                    raise TelloException("Failed to grab video feed after 2 tries! Try restarting the script.")
                else:
                    print("Failed to get video feed, trying again...")
                    continue
            break

        threading.Thread(target=self._frame_updater).start()

    def _frame_updater(self):
        while True:
            try:
                for frame in self.container.decode(video=0):
                    self.frame = np.array(frame.to_image())

                    if not self.running:
                        self.tello.streamoff()
                        self.container.close()
                        break
            except av.error.ExitError:
                continue
            except av.error.InvalidDataError:
                continue

    def stop(self):
        self.running = False

    def get_frame(self):
        return self.frame