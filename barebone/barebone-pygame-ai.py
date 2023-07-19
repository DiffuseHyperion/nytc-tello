from djitellopy import Tello
from VideoFeed import VideoFeed
import VideoFeedInterpreter
import time
import cv2
import os
import pygame
import numpy as np

pygame.init()


class VideoThread:

    def __init__(self, tello_object):
        self.running = True
        self.tello = tello_object
        self.screen = pygame.display.set_mode((960, 720))
        self.video_feed = VideoFeed(tello_object)
        self.video_feed_interpreter = VideoFeedInterpreter.VideoFeedIntepreter(self.video_feed,
                                                                               os.path.join(os.getcwd(), "..", "models",
                                                                                            "mobilenetv1"),
                                                                               0.5)

    def _add_batt_text(self, frame):
        text = "Battery: " + str(self.tello.get_battery()) + "%"
        cv2.putText(frame, text, (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.video_feed.stop()
                    self.tello.end()
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = True
                        self.video_feed.stop()
                        self.tello.end()
                        break

            self.screen.fill(([0, 0, 0]))
            frame = self.video_feed_interpreter.get_frame()
            self._add_batt_text(frame)

            frame = np.rot90(frame)
            frame = np.flipud(frame)
            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.flip()


tello = Tello()
tello.connect()
video = VideoThread(tello)
time.sleep(1)
video.run()
