from djitellopy import Tello
from VideoFeed import VideoFeed
import threading
import cv2


class VideoThread:

    def __init__(self, tello_object):
        self.running = True
        self.tello = tello_object
        self.video_feed = VideoFeed(tello_object)
        threading.Thread(target=self._video).start()

    def _add_batt_text(self, frame):
        text = "Battery: " + str(self.tello.get_battery()) + "%"
        cv2.putText(frame, text, (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def _video(self):
        frame_skip = 300
        while self.running:
            if frame_skip >= 0:
                frame_skip -= 1
                if frame_skip < 0:
                    print("Starting video!")
                continue

            frame = self.video_feed.get_frame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._add_batt_text(frame)
            cv2.imshow("Video", frame)

            height, width = frame.shape[:2]
            print("Height: {}, Width: {}".format(height, width))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                # self.tello.land()
                self.tello.end()
                self.video_feed.stop()
                self.running = False


tello = Tello()
tello.connect()
video = VideoThread(tello)
