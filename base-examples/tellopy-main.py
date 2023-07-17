from tellopy import Tello
import cv2
import threading
import keyboard


class FrontEnd(object):
    def __init__(self, speed):
        # Handles setting up fields within class, initializing drone settings and initializing pygame control panel
        print("Initializing control panel...")
        # Setup fields
        self.running = True
        self.speed = speed
        self.frame_read = None

        print("Initializing drone...")
        # Init Tello
        self.tello = Tello()
        self.tello.connect()
        #self.tello.set_speed(self.speed)
        #self.tello.set_video_direction(Tello.CAMERA_FORWARD)
        print("Finished initializing drone.")

        print("Creating event thread...")
        # Initialize event thread
        event_thread = threading.Thread(target=self._handle_event_thread)
        event_thread.start()
        print("Finished starting event thread.")

        print("Getting video camera feed...")
        # Start getting BackgroundFrameRead object in separate async thread because it takes a while (for some reason)
        self._get_frame_reader()
        print("Found video camera feed!")

        print("Creating UI thread...")
        # Initialize UI updating thread
        ui_thread = threading.Thread(target=self._update_ui_thread)
        ui_thread.start()
        print("Finished starting UI thread.")

        print("Finished initializing control panel!")

    def _stop_session(self):
        self.running = False
        self._quit()
        self.tello.end()
        cv2.destroyAllWindows()

    def _get_frame_reader(self):
        self.tello.streamoff()  # In case streaming is on. This happens when we quit this program without the escape key.
        self.tello.streamon()
        tries = 0
        while True:
            try:
                print("Waiting for video camera to start...")
                self.frame_read = self.tello.get_frame_read()  # Sometimes this fails for no reason. Putting this in a loop allows us to just retry if it fails
            except TelloException:
                tries += 1
                if tries >= 3:
                    print("Could not get drone video feed after 3 tries. Exiting.")
                    exit()
                else:
                    print("Failed to get drone video feed. Retrying...")
                    continue
            break

    def _update_ui_thread(self):
        while self.running:
            try:
                # Update UI with Tello frames
                self.current_frame = self.frame_read.frame

                battery_frame = self.current_frame.copy()
                text = "Battery: " + str(self.tello.get_battery()) + "%"
                cv2.putText(battery_frame, text, (5, 720 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Control Panel", battery_frame)

                contour_frame = self.current_frame.copy()
                gray_frame = cv2.cvtColor(contour_frame, cv2.COLOR_BGR2GRAY)
                canny_frame = cv2.Canny(gray_frame, 30, 200)
                contours, hierarchy = cv2.findContours(canny_frame.copy(),
                                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 3)

                cv2.imshow("Gray", gray_frame)
                cv2.imshow("Canny", canny_frame)
                cv2.imshow('Contours', contour_frame)
                cv2.waitKey(1)
            except TelloException:
                continue

    def _handle_event_thread(self):
        self.old_battery = self.tello.get_battery()
        while self.running:
            if keyboard.is_pressed("escape"):
                self._stop_session()
                break

            battery = self.tello.get_battery()
            if self.old_battery != battery:
                self._battery_changed(battery)
                self.old_battery = battery

    def run(self):
        print("Running main code now...")
        self._main()
        print("Main code finished!")
        self._stop_session()

    """
    Main code goes here!
    """

    def _main(self):
        self.tello.takeoff()
        self.tello.land()

    """
    Battery update callback
    Called when battery level changes
    """

    def _battery_changed(self, level):
        if level <= 10:
            print("Battery level critically low. Immediately stopping!")
            self._stop_session()

    """
    Quit callback
    Called once session is gracefully terminated (via pygame window)
    Will not be called if user forcefully terminates using console.
    """

    def _quit(self):
        pass


if __name__ == '__main__':
    frontend = FrontEnd(10)
    # frontend.run()
