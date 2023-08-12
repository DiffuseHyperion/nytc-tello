from djitellopy import Tello
import cv2
import pygame
import numpy as np
import time

### CUSTOM SECTION ###

from sys import platform  # Used to detect computer platform (Windows, Linux, etc)
from VideoRead import VideoRead  # Used to get frames from video camera, see line 116
import threading  # Used to run code concurrently with each other, see line 202

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
FPS = 120
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

        # Enable sending commands to the drone
        self.tello.connect()

        # Set speed of drone, moved up here because nicer
        self.tello.set_speed(self.speed)

        # VideoRead is my custom method of getting frames from the drone's video feed
        # Did this because the method in djitellopy kinna sucks and crashes when there are no more frames from the video feed
        # My method simply waits and retry getting frames later instead
        self.frame_read = VideoRead(self.tello)
        ### END OF CUSTOM SECTION ###

    def _process_image(self, original_frame: np.ndarray):
        # Read and resize image
        original_shape = np.shape(original_frame)
        input_shape = self.input_details[0]['shape']
        new_image = cv2.resize(original_frame, (input_shape[1], input_shape[2]))

        self.interpreter.set_tensor(self.input_details[0]['index'], [new_image])
        self.interpreter.invoke()

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
                cv2.putText(new_image, img_text, (int(unnormed_coords[1]), int(unnormed_coords[0]) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                break

            # counter = 0

        new_image = cv2.resize(new_image, (original_shape[1], original_shape[0]))

        return new_image

    def _post_process_image(self, image: np.ndarray, stats: bool, flipped: bool, colour: bool):
        if stats:
            # Display battery
            text = "Battery: {}%".format(self.tello.get_battery())
            cv2.putText(image, text, (5, FRAME_HEIGHT - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display last snapshot timing
            text = "Last snapshot: {}".format(self.last_snapshot)
            cv2.putText(image, text, (5, FRAME_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, self.text_color, 2)
        if flipped:
            image = np.rot90(image)
            image = np.flipud(image)
        if colour:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

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

            frame = self._post_process_image(self.frame_read.get_frame(), True, True, False)
            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / FPS)
        ### START OF CUSTOM SECTION ###
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

            def process():
                processed_image = self._post_process_image(self._process_image(self.frame_read.get_frame()),
                                                           True,
                                                           False,
                                                           True)
                cv2.imshow("Screenshot", processed_image)

                # Press Enter to take picture with bounding box
                cv2.imwrite(f"picture-{self.last_snapshot}.png",
                            cv2.UMat(processed_image))
                cv2.waitKey(0)

            threading.Thread(target=process).start()

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
