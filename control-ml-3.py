"""
An incredibly over engineered solution to the long detection time.
Video thread will now contain a deque (a two way list, you can add from the top and bottom).
Every tick (dictated by fps), a future (an object that returns a value later) will be added onto the back of the deque.
Then, a thread will be created to run object detection on the current frame,
and pass the result frame to the future.

The video thread will now wait for the top future to finish before it displays that frame
and removes the future from the deque, waiting for the next frame.

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
from asyncio import Future # This is used for the frame queue system.

BUFFERED_FRAMES = 5 # How many frames in start object detection in advance. See explanation image for details.
# This should vary based on available threads/cpu cores on the host system.

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

        # Import labels 
        with open(label_path, "r") as f:
            self.labels = [line.strip() for line in f.readlines()]
            f.close()

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
        ### END OF CUSTOM SECTION ###

    ### START OF CUSTOM SECTION ###
    # Method for event_thread, see method run()
    def event_routine(self):
        print("start event")
    ### END OF CUSTOM SECTION ###
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

    ### START OF CUSTOM SECTION ###
    # Handles the frame queue system.
    # Every frame (dictated by FPS constant), this method will create a Future and a video_routine thread,
    # and add it into a deque. Then, it checks if the oldest frame has finished object detection.
    # If it has, it will show the produced frame and remove it from the queue, then start appropriate threads
    # which got pushed into the "buffer zone". This way, delays between each frame found in v2 is minimized as
    # object detection was done beforehand, rather than on the spot.
    def video_manager(self):
        # Queue: NEWEST FRAME -----> OLDEST FRAME
        queue = deque()
        while not self.should_stop:
            # Queue next frame
            future_frame = Future()
            future_thread = \
                threading.Thread(target=self.video_routine, args=[future_frame, self.frame_read.get_frame()])
            queue.appendleft((future_frame, future_thread))
            if len(queue) <= BUFFERED_FRAMES:
                future_thread.start()

            # Query incoming frame
            incoming_frame: Future = queue[0][0]
            if incoming_frame.done():
                # Show incoming frame
                new_image = queue.pop()
                new_image = np.rot90(new_image)
                new_image = np.flipud(new_image)

                frame = pygame.surfarray.make_surface(new_image)
                self.screen.blit(frame, (0, 0))
                pygame.display.update()

                # Start object detection on appropriate thread
                if len(queue) >= BUFFERED_FRAMES:
                    incoming_thread: threading.Thread = queue[BUFFERED_FRAMES - 1][1]
                    incoming_thread.start()
            time.sleep(1 / FPS)
    # Method for video_thread, see method run()
    def video_routine(self, future: Future, frame: np.ndarray):
        ### END OF CUSTOM SECTION ###

        # Read and resize image
        original_shape = np.shape(frame)
        input_shape = self.input_details[0]['shape']
        new_image = cv2.resize(frame, (input_shape[1], input_shape[2]))

        self.interpreter.set_tensor(self.input_details[0]['index'], [new_image])
        start_time = time.time()
        self.interpreter.invoke()
        time_taken = time.time() - start_time
        print("Took " + str(time_taken) + " seconds")

        boxes = self.interpreter.get_tensor(self.output_details[0]['index']).squeeze()
        classes = self.interpreter.get_tensor(self.output_details[1]['index']).squeeze()
        scores = self.interpreter.get_tensor(self.output_details[2]['index']).squeeze()

        for i in range(len(scores)):
            if scores[i] > confidence_threshold:
                # Unnormalize boundaries
                unnormed_coords = boxes[i] * input_shape[1]
                start_point = (int(unnormed_coords[1]), int(unnormed_coords[0]))
                end_point = (int(unnormed_coords[3]), int(unnormed_coords[2]))
                # Draw bounding box
                drawn = cv2.rectangle(new_image, start_point, end_point, color=(0, 255, 0), thickness=2)
                # Add label and score
                img_text = f"{self.labels[int(classes[i])]}: {scores[i]:.3f}"
                output_label = cv2.putText(new_image, img_text,
                                           (int(unnormed_coords[1]), int(unnormed_coords[0]) + 15),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                break

            # counter = 0

        new_image = cv2.resize(new_image, (original_shape[1], original_shape[0]))
        # Display battery
        text = "Battery: {}%".format(self.tello.get_battery())
        cv2.putText(new_image, text, (5, FRAME_HEIGHT - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display last snapshot timing
        text = "Last snapshot: {}".format(self.last_snapshot)
        cv2.putText(new_image, text, (5, FRAME_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, self.text_color, 2)

        ### START OF CUSTOM SECTION ###
        future.set_result(new_image)
        ### END OF CUSTOM SECTION ###

    def run(self):
        ### START OF CUSTOM SECTION ###
        print("start run")
        # Create video and event threads
        # Threads allow code to run with other code at the same time, without both code blocking each other.
        event_thread = threading.Thread(target=self.event_routine)
        video_thread = threading.Thread(target=self.video_manager())

        # Start threads
        event_thread.start()
        video_thread.start()

        # Wait for both threads to finish, in this case when the code is stopped by event_thread
        event_thread.join()
        video_thread.join()

        # Stop video feed
        self.frame_read.stop()
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
