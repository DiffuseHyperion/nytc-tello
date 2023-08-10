"""
An incredibly overengineered solution to the long detection time.
Video thread will now contain a deque (a two way list, you can add from the top and bottom).
Every tick (dictated by fps), a future (an object that returns a value later) will be added onto the back of the deque.

Pros:
Theoretically perfect framerate
More frames shown, as its no longer blocked by existing frame object detection
Cons:
Significantly more resource intensive
Frames will have a noticeable latency
Screenshots might be wonky
"""

from djitellopy import Tello
import cv2
import pygame
import numpy as np
import time

### CUSTOM SECTION ###

from sys import platform  # Used to detect computer platform (Windows, Linux, etc)

from VideoRead import VideoRead  # Used to get frames from video camera, see line 116
import threading  # Used to run code concurrently with each other, see line 202

from collections import deque # Deque stands for "double-ended queue",
# basically a list but you can add items to it from both ends. This is used for frame queue system.
from FrameInterpreterManager import FrameInterpreterManager

BUFFERED_FRAMES = 5 # How many frames in start object detection in advance.

# This ensures compatability with Windows PCs because tflite_runtime does not exist for Windows
if platform == "linux" or platform == "linux2":
    from tflite_runtime.interpreter import Interpreter
elif platform == "win32":
    from tensorflow.lite.python.interpreter import Interpreter

# This makes it easier to swap between models
loop = True
base_path = ""
while loop:
    model = input("Enter the model you want to use: ([c]ircle/[l]ego/[sample]/[s]hapes): ")
    base_path = "models/"
    if model == "c" or model == "circle":
        base_path += "circle/"
    elif model == "l" or model == "lego":
        base_path += "lego/"
    elif model == "sample":
        base_path += "sample/"
    elif model == "s" or model == "shapes":
        base_path += "shapes/"
    else:
        print("Unknown model entered! Please try again...")
        continue
    loop = False

label_path = base_path + "dict.txt"
model_path = base_path + "model.tflite"
confidence_threshold = 0.5

### END OF CUSTOM SECTION ###

# Speed of the drone
S = 60
# Frames per second of the pygame window display
# A low number results in input lag, as input information is processed once per frame
FPS = 60
# Settings for frame size
FRAME_WIDTH = 960
FRAME_HEIGHT = 660


class FrontEnd(object):
    """
        Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        Press enter key to take a snapshot.
        The controls are:
        - T: Takeoff
        - L: Land
        - Arrow keys: Forward, backward, left, right
        - A and D: Counter clockwise, clockwise rotations (yaw)
        - W and S: Up, down
    """

    def __init__(self):
        # Init pygame
        pygame.init()

        # Create pygame window 
        pygame.display.set_caption("Tello video stream")
        # Set width and height
        self.screen = pygame.display.set_mode([FRAME_WIDTH, FRAME_HEIGHT])

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        # Initialize snapshot time 
        self.last_snapshot = "N/A"
        self.text_color = (0, 0, 255)

        self.send_rc_control = False

        # Initialize TFLite model and allocate tensors
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details, self.output_details = self.interpreter.get_input_details(), self.interpreter.get_output_details()

        # Create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // FPS)

        ### CUSTOM SECTION ###
        # Allows for event_routine and video_routine to communicate between each other when to stop
        self.should_stop = False

        # Allows video_routine to share finalized frames to screenshot function
        self.new_image = None

        # Enable sending commands to the drone
        self.tello.connect()

        # Set speed of drone, moved up here because nicer
        self.tello.set_speed(self.speed)

        # VideoRead is my custom method of getting frames from the drone's video feed
        # Did this because the method in djitellopy kinna sucks and crashes when there are no more frames from the video feed
        # My method simply waits and retry getting frames later instead
        self.frame_read = VideoRead(self.tello)

        self.frame_manager = FrameInterpreterManager(5, model_path, label_path, 0.5, self.frame_read, FPS)
        ### END OF CUSTOM SECTION ###

    def run(self):
        while not self.should_stop:
            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.update()
                elif event.type == pygame.QUIT:
                    self.should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)

            self.screen.fill([0, 0, 0])

            outgoing_queue: deque = self.frame_manager.get_outgoing_queue()
            if len(outgoing_queue) <= 0:
                continue
            print("Showing outgoing frame!")
            outgoing_frame: np.ndarray = outgoing_queue.pop()
            frame = pygame.surfarray.make_surface(outgoing_frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / FPS)
        ### START OF CUSTOM SECTION ###
        # Stop video feed
        self.frame_read.stop()
        self.frame_manager.stop()
        ### END OF CUSTOM SECTION ###
        # Call it always before finishing. To deallocate resources.
        self.tello.end()

    def keydown(self, key):
        """
            Update velocities based on key pressed
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = S
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -S
        elif key == pygame.K_a:  # set yaw counter clockwise velocity
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # set yaw clockwise velocity
            self.yaw_velocity = S
        elif key == pygame.K_RETURN:
            # Update "Last snapshot time" label
            t = time.localtime()
            self.last_snapshot = time.strftime("%H:%M:%S", t)
            if self.text_color == (0, 0, 255):
                self.text_color = (255, 0, 0)
            else:
                self.text_color = (0, 0, 255)

            # Press Enter to take picture with bounding box
            cv2.imwrite(f"picture-{self.last_snapshot}.png", cv2.cvtColor(self.new_image, cv2.COLOR_BGR2RGB))

    def keyup(self, key):
        """
            Update velocities based on key released
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            ### START OF CUSTOM SECTION ###
            # Removed "not" from the start of this line, idk why it was there???
            self.tello.land()
            ### END OF CUSTOM SECTION ###
            self.send_rc_control = False

    def update(self):
        """
            Update routine. Send velocities to Tello.
        """
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)


def main():
    frontend = FrontEnd()

    # Run frontend
    frontend.run()


if __name__ == '__main__':
    main()
