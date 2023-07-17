"""
Initial project. Adapted from Google's pygame example, stripped down to serve as a "control panel".
Abandoned due to massive latency issues and too much complexity.
Barebone versions should be used instead.
"""
from djitellopy import Tello, TelloException
import cv2
import threading
import keyboard

SPEED = 10
class FrontEnd(object):
    def __init__(self):
        # Handles setting up fields within class, initializing drone settings and initializing pygame control panel
        print("Creating initialize field thread...")
        field_thread = threading.Thread(target=self._init_fields)
        field_thread.start()
        print("Finished starting initialize fields thread.")

        print("Creating initialize drone thread...")
        drone_thread = threading.Thread(target=self._init_drone)
        drone_thread.start()
        print("Finished starting initialize drone thread.")

        print("Waiting for drone to finish initializing...")
        drone_thread.join()
        print("Creating event thread...")
        event_thread = threading.Thread(target=self._event_thread)
        event_thread.start()
        print("Finished starting event thread.")

        print("Creating video feed thread...")
        frame_thread = threading.Thread(target=self._get_frame_read_thread)
        frame_thread.start()
        print("Finished starting video feed thread.")

        print("Waiting for video feed...")
        frame_thread.join()
        print("Creating UI thread...")
        ui_thread = threading.Thread(target=self._update_ui_thread)
        ui_thread.start()
        print("Finished starting UI thread.")

        print("Finished initializing control panel!")

    def _init_fields(self):
        self.running = True
        self.ready = False
        self.speed = SPEED
        self.frame_read = None
        self.tello = None
        print("Finished initializing fields.")

    def _init_drone(self):
        while True:
            try:
                self.tello = Tello()
                self.tello.connect()
                self.tello.set_speed(self.speed)
                self.tello.set_video_direction(Tello.CAMERA_FORWARD)
                self.tello.streamoff()
            except TelloException:
                print("Failed to initialize drone. Retrying...")
                continue
            except KeyError:
                print("Failed to initialize drone. Retrying...")
                continue
            break
        print("Finished initializing drone.")

    def _stop_session(self):
        self.running = False
        self._quit()
        self.tello.streamoff()
        self.tello.end()
        cv2.destroyAllWindows()
        print("Goodbye!")

    def _get_frame_read_thread(self):
        self.tello.streamon()
        while True:
            try:
                print("Waiting for video camera to start...")
                self.frame_read = self.tello.get_frame_read()  # Sometimes this fails for no reason. Putting this in a loop allows us to just retry if it fails
            except TelloException:
                print("Failed to get drone video feed. Retrying...")
                continue
            break
        print("Finished getting video feed!")

    def _update_ui_thread(self):
        while self.running:
            if self.frame_read.stopped:
                break
            try:
                frame = self.frame_read.frame
                # Update UI with Tello frames
                text = "Battery: " + str(self.tello.get_battery()) + "%"
                cv2.putText(frame, text, (5, 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Control Panel", frame)
                cv2.waitKey(1)
                """
                    contour_frame = self.current_frame.copy()
                    gray_frame = cv2.cvtColor(contour_frame, cv2.COLOR_BGR2GRAY)
                    canny_frame = cv2.Canny(gray_frame, 30, 200)
                    contours, hierarchy = cv2.findContours(canny_frame.copy(),
                                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 3)

                    cv2.imshow("Gray", gray_frame)
                    cv2.imshow("Canny", canny_frame)
                    cv2.imshow('Contours', contour_frame)
                """
            except TelloException:
                continue

    def _event_thread(self):
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
    frontend = FrontEnd()
    # frontend.run()
