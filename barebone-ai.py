from djitellopy import Tello
from VideoFeed import VideoFeed
import VideoFeedInterpreter
import threading
import cv2
import os


class VideoThread:

    def __init__(self, tello_object):
        self.running = True
        self.tello = tello_object
        self.video_feed = VideoFeed(tello_object)
        self.video_feed_interpreter = VideoFeedInterpreter.VideoFeedIntepreter(self.video_feed,
                                                                               os.path.join(os.getcwd(), "models",
                                                                                            "mobilenet"),
                                                                               0.5, VideoFeedInterpreter.TF1_MODEL)

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

            ai_frame = self.video_feed_interpreter.get_frame()
            ai_frame = cv2.cvtColor(ai_frame, cv2.COLOR_BGR2RGB)
            self._add_batt_text(ai_frame)
            cv2.imshow("Video", ai_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopping video!")
                # self.tello.land()
                self.tello.end()
                self.running = False
        print("Thread ending!")


tello = Tello()
tello.connect()
video = VideoThread(tello)
