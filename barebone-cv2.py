import time

import cv2
from djitellopy import Tello

from VideoFeed import VideoFeed


class VideoThread:

    def __init__(self, tello_object):
        self.running = True
        self.tello = tello_object
        self.video_feed = VideoFeed(tello_object)

    def _add_batt_text(self, frame):
        text = "Battery: " + str(self.tello.get_battery()) + "%"
        cv2.putText(frame, text, (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def run(self):
        cv2.startWindowThread()
        while self.running:
            print("start loop")
            frame = self.video_feed.get_frame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._add_batt_text(frame)
            print("Showing frame!")
            cv2.imshow("Video", frame) # Hangs here
            print("Frame shown")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            if cv2.waitKey(0) == ord("q"):
                # self.tello.land()
                self.tello.end()
                self.video_feed.stop()
                cv2.destroyAllWindows()
                self.running = False
            print("end loop")


tello = Tello()
tello.connect()
video = VideoThread(tello)
time.sleep(1)
video.run()
