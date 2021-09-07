import cv2
from Track_utils import  *
import time
import numpy as np
from pynput.keyboard import Controller as kC
from pynput.mouse import Controller as mC
import pynput

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
screen_resolution = np.array([1280, 720])

handdetector = HandDetector(maxHands=2, detectionCon=0.75, minTrackCon=0.75)
facedetector = FaceDetector(minDetectionCon=0.8)
keyboard = kC()
mouse = mC()
count = 0
start_time = time.time() # To calculate fps
mean_time = time.time() # To smooth key
keys_press = {'Q':0, 'W':0, 'E':0, 'R':0, 'D':0}
original_key_press = keys_press.copy()
text_flag = -1
old_flag = 0

class Button(object):
    def __init__(self, size, img_size):
        self.size_width, self.size_height = size # absolute size
        self.half_size_w, self.half_size_h = int(self.size_width / 2), int(self.size_height / 2)
        self.img_width, self.img_height = img_size
        self.row_numbers, self.column_numbers = 0.7 * self.img_width // self.size_width, 0.7 * self.img_height // self.size_height
        self.buttons_pos = {}

    def draw(self, start_location, texts, img):
        original_location = start_location.copy()
        number_texts = len(texts)
        assert number_texts <= self.row_numbers * self.column_numbers, 'Too many texts please reduce number or text size'
        for i in range(number_texts):
            start_location = [original_location[0], int(original_location[1] + i * self.size_height * 1.1)] # 下一行
            for j in range(len(texts[i])):
                x, y = start_location
                x_left, y_left, x_right, y_right = x-self.half_size_w, y-self.half_size_h, x+self.half_size_w, y+self.half_size_h
                cv2.rectangle(img, (x_left, y_left),
                              (x_right, y_right), color=(200, 120, 255), thickness=cv2.FILLED)
                cv2.putText(img, str(texts[i][j]), [int(x_left + 0.3 * self.size_width), int(y_left + 0.6 * self.size_height)], cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(255, 255, 255), thickness=4)
                location_row, location_column  = int(x + self.size_width * 1.1), y # next button
                start_location = [location_row, location_column]

                self.buttons_pos[texts[i][j]] = ([x_left, y_left, x_right, y_right]) # hold all positions of buttons
        return img

button = Button([50, 70], [1280, 720]) # size and image size
texts_left = ['R', 'E', 'W', 'Q', 'D']
texts = [list(range(7))]
texts.append(['R', 'E', 'Q', 'W', 'S'])
texts.append(['A', 'S', 'D', 'F', 'G'])
texts.append(['B'])

start_location = [70, 90]
left_old_old = np.zeros((5, 2))
left_old = np.zeros((5, 2))
right_old = np.array([0, 0])

while True:
    count += 1
    flag, capture = cap.read()
    picture = cv2.flip(capture, 1)  # flip the screen to match
    hand_new = handdetector.findHands(picture)
    coor_list, bbox = handdetector.findPosition(picture)
    face_new, _ = facedetector.findFaces(hand_new, draw=False)
    c = cv2.waitKey(1)
    if len(coor_list) == 2: # get left and right hands simultaneously
        coor_list_right = coor_list[0]
        coor_list_left = coor_list[1]
        five_left, five_right = finger_position(coor_list_left, coor_list_right)
        if not left_old.all() or not left_old_old.all(): # all zero to avoid initial situation, hands move in the screen
            left_old_old = left_old
            left_old = five_left
            continue
        d_distance_old = compute_dx_dy(left_old_old, left_old)
        d_distance = compute_dx_dy(left_old, five_left)
        dy_old, dy = d_distance_old[:, 1], d_distance[:, 1]
        dy_old[0], dy[0] = d_distance_old[0, 0], d_distance[0, 0]
        metric_distance = dy_old * dy    # must be negative and find most negative key
        get_metric = (metric_distance < 0) * metric_distance
        if not get_metric.all():  finger_index = np.argmin(get_metric)
        else: finger_index = None
        # print(get_metric)
        if finger_index is not None and np.min(get_metric) < -100 and np.max(get_metric) == 0: # avoid too fast or small oscillation make unintended activation
            if time.time() - mean_time > 0.2:
                keyboard.press(texts_left[finger_index])
                print(texts_left[finger_index])
                mean_time = time.time()
        left_old_old, left_old = left_old, five_left # update old position

        position = ((five_right[1, :] + np.array([0, 250]) - screen_resolution * 0.5) / (screen_resolution * 0.5)) * np.array([2560, 1200])
        if right_old.all(): old_flag = np.isclose(euclidean(position), euclidean(right_old), atol=10, rtol=0)
        if old_flag: position = right_old.copy(); old_flag = 0
        else: right_old = position.copy()
        mouse.position = position

        # This is the other way to control mouse(relative)
        # position = ((five_right[1, :] + np.array([0, 250]) - screen_resolution * 0.5) / (screen_resolution * 0.5)) * np.array([2560, 1200])
        # if right_old.all(): old_flag = np.isclose(euclidean(position), euclidean(right_old), atol=25, rtol=0)
        # if old_flag: movement = np.array([0, 0]); old_flag = 0
        # else: movement = position - right_old; right_old = position.copy()
        # mouse.move(movement[0], movement[1])

        angle = thumb_angle(coor_list_right)
        # print(angle)
        if angle < 150: mouse.press(pynput.mouse.Button.left)
        else: mouse.release(pynput.mouse.Button.left)

    if c == 27:
        break
    if c in [ord('z'), ord('Z')]: # choose button, show fps
        print('Open/Close fps')
        text_flag = -1 * text_flag
    if text_flag > 0:
        real_fps = count / (time.time() - start_time)
        cv2.putText(face_new, 'FPS: %.2f'%(real_fps), (100, 160), cv2.FONT_HERSHEY_DUPLEX, 1, color=(200, 120, 255), lineType=3)
        count, start_time = 0, time.time()
    # key_board_img = button.draw(start_location, texts, face_new)
    cv2.imshow('camera', face_new)

cap.release()
cv2.destroyAllWindows()