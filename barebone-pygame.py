from djitellopy import Tello
import threading
import cv2
import numpy as np
import pygame

from VideoFeed import VideoFeed

pygame.init()


# abandoned due to massive consistency issues and horrible video latency (~3 seconds)
# cv2 should be used instead

class VideoThread:

    def __init__(self, tello_object):
        self.running = True
        self.tello = tello_object
        self.screen = pygame.display.set_mode((720, 720))

        self.videofeed = VideoFeed(tello_object)

        threading.Thread(target=self._main_manager_thread).start()
        threading.Thread(target=self._event_thread).start()

    def _main_manager_thread(self):
        frame_skip = 300
        while self.running:
            if frame_skip >= 0:
                frame_skip -= 1
                if frame_skip < 0:
                    print("Starting video!")
                continue

            self.screen.fill([0, 0, 0])

            frame = self.videofeed.frame
            text = "Battery: " + str(self.tello.get_battery()) + "%"
            cv2.putText(frame, text, (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            frame = np.rot90(frame)
            frame = np.flipud(frame)
            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

    def _event_thread(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False


tello = Tello()
tello.connect()
video = VideoThread(tello)
