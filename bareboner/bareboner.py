"""
An even more barebone version of the tello script. Should be used for trying out new stuff.
"""
from djitellopy import Tello
import keyboard
import threading


def main(tello):
    """
        tello.enable_mission_pads()
        pad = tello.get_mission_pad_id()
        print(pad)

        while pad != 2:
            tello.move_forward(50)
            pad = tello.get_mission_pad_id()
            print(pad)
    """
    tello.takeoff()
    tello.land()


def handle_stop(tello):
    while True:
        if keyboard.is_pressed("escape"):
            print("emergency stop!")
            tello.land()
            tello.end()
            exit()


tello = Tello()
tello.connect()
main_thread = threading.Thread(target=main, args=(tello,))
main_thread.start()

keyboard_thread = threading.Thread(target=handle_stop, args=(tello,))
keyboard_thread.start()
