import cv2
import threading
import matplotlib.pyplot as plt

pic = cv2.imread("yippee.jpg")
plt.imshow(pic)
plt.show()

""" Working example
def test():
    pic = cv2.imread("yippee.jpg")
    cv2.imshow("test", pic)
    cv2.waitKey(0)


"""

""" Working example
class test:
    def __init__(self):
        threading.Thread(target=self.test).start()

    def test(self):
        pic = cv2.imread("yippee.jpg")
        cv2.imshow("test", pic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


test = test()
"""

