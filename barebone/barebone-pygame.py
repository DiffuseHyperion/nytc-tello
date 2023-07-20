import time
import cv2
import pygame
import numpy as np
from djitellopy import Tello

from VideoFeed import VideoFeed

pygame.init()


class VideoThread:

    def __init__(self, tello_object):
        self.running = True
        self.tello = tello_object
        self.screen = pygame.display.set_mode((960, 720))
        self.video_feed = VideoFeed(tello_object)

    def _add_batt_text(self, frame):
        text = "Battery: " + str(self.tello.get_battery()) + "%"
        cv2.putText(frame, text, (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    self.tello.end()
                    self.video_feed.stop()
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = True
                        self.tello.end()
                        self.video_feed.stop()
                        break

            self.screen.fill(([0, 0, 0]))
            frame = self.video_feed.get_frame()
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
