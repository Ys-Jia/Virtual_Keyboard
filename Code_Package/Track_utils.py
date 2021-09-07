import cv2
import mediapipe as mp
import math
import numpy as np

class HandDetector:
    """
    Finds Hands using the mediapipe library. Exports the landmarks
    in pixel format. Adds extra functionalities like finding how
    many fingers are up or the distance between two fingers. Also
    provides bounding box info of the hand found.
    """

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):
        """
        :param mode: In static mode, detection is done on each image: slower
        :param maxHands: Maximum number of hands to detect
        :param detectionCon: Minimum Detection Confidence Threshold
        :param minTrackCon: Minimum Tracking Confidence Threshold
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.lmList = []

    def findHands(self, img, draw=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        """
        Finds landmarks of a single hand and puts them in a list
        in pixel format. Also finds the bounding box around the hand.

        :param img: main image to find hand in
        :param draw: Flag to draw the output on the image.
        :return: list of landmarks in pixel format; bounding box
        """

        bbox = []
        self.bboxInfo = {}
        self.lmList = {}
        self.fake_List = {} # fake use to document, real use to output(to adjust the order or left and right)
        self.fake_bboxInfo = {}
        if self.results.multi_hand_landmarks:
            # myHand = self.results.multi_hand_landmarks[handNo]
            for hand_index in range(len(self.results.multi_hand_landmarks)):
                nowHand = self.results.multi_hand_landmarks[hand_index]
                fake_list, xList, yList = [], [], []
                for id, lm in enumerate(nowHand.landmark):
                    h, w, c = img.shape
                    px, py = int(lm.x * w), int(lm.y * h)
                    xList.append(px)
                    yList.append(py)
                    fake_list.append([px, py])
                    if draw:
                        cv2.circle(img, (px, py), 5, (min(50*hand_index, 255), 50, min(50*hand_index, 255)), cv2.FILLED)
                self.fake_List[hand_index] = fake_list
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)
                self.fake_bboxInfo.update({hand_index: (bbox, (cx, cy))})

            hands_Type = self.handType()
            if len(hands_Type) == 2: # check it should be 0-->Right, 1-->Left
                self.lmList[0], self.lmList[1] = self.fake_List[hands_Type.index('Right')], self.fake_List[hands_Type.index('Left')]
                self.bboxInfo[0], self.bboxInfo[1] = self.fake_bboxInfo[hands_Type.index('Right')], self.fake_bboxInfo[hands_Type.index('Left')]
                self.fake_List, self.fake_bboxInfo = self.lmList.copy(), self.bboxInfo.copy()
                hands_Type = ['Right', 'Left']
            else:
                self.lmList = self.fake_List.copy()
                self.bboxInfo = self.fake_bboxInfo.copy()

            if draw:
                for hand_index in self.bboxInfo:
                    bbox = self.bboxInfo[hand_index][0]
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (0, 255, 0), 2)
                    cv2.putText(img, hands_Type[hand_index]+str(hand_index), (bbox[0] - 20, bbox[1] + 20), cv2.FONT_HERSHEY_DUPLEX, 1,
                                color=(200, 120, 255), lineType=3)  # 要每张图都加字才行！

        return self.lmList, self.bboxInfo

    def fingersUp(self):
        """
        Finds how many fingers are open and returns in a list.
        Considers left and right hands separately
        :return: List of which fingers are up
        """
        if self.results.multi_hand_landmarks:
            myHandTypes = self.handType()
            fingers = []
            up_fingers = {}
            # Thumb
            for hand_index in range(len(myHandTypes)):
                myHandType = myHandTypes[hand_index]
                if myHandType == "Right":
                    if self.lmList[hand_index][self.tipIds[0]][0] > self.lmList[hand_index][self.tipIds[0] - 1][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    if self.lmList[hand_index][self.tipIds[0]][0] < self.lmList[hand_index][self.tipIds[0] - 1][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

            # 4 Fingers
                for id in range(1, 5):
                    if self.lmList[hand_index][self.tipIds[id]][1] < self.lmList[hand_index][self.tipIds[id] - 2][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                up_fingers[hand_index] = fingers
            return up_fingers

    def findDistance(self, handNo, p1, p2, img, draw=True):
        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1 - Index of Landmark 1.
        :param p2: Point2 - Index of Landmark 2.
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                 Image with output drawn
                 Line information
        """

        if self.results.multi_hand_landmarks:

            x1, y1 = self.lmList[handNo][p1][0], self.lmList[handNo][p1][1]
            x2, y2 = self.lmList[handNo][p2][0], self.lmList[handNo][p2][1]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if draw:
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)
            return length, img, [x1, y1, x2, y2, cx, cy]

    def handType(self):
        """
        Checks if the hand is left or right
        :return: "Right" or "Left"
        """
        if self.results.multi_hand_landmarks:
            number_hands = len(self.fake_List)
            type_results = []
            if number_hands == 2:
                if self.fake_bboxInfo[0][1][0] < self.fake_bboxInfo[1][1][0]: return ['Left', 'Right'] # x坐标小于的则是左手
                else: return ['Right', 'Left']
            else:
                for i in range(number_hands):
                    type_hand = self.handType_single(handNo=i)
                    type_results.append(type_hand)
                return type_results

    def handType_single(self, handNo):
        if self.fake_List[handNo][17][0] < self.fake_List[handNo][5][0]:
            return "Left"
        else:
            return "Right"

class FaceDetector:
    """
    Find faces in realtime using the light weight model provided in the mediapipe
    library.
    """

    def __init__(self, minDetectionCon=0.5):
        """
        :param minDetectionCon: Minimum Detection Confidence Threshold
        """

        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        """
        Find faces in an image and return the bbox info
        :param img: Image to find the faces in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings.
                 Bounding Box list.
        """

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)
                bboxInfo = {"id": id, "bbox": bbox, "score": detection.score, "center": (cx, cy)}
                bboxs.append(bboxInfo)
                if draw:
                    img = cv2.rectangle(img, bbox, (255, 0, 255), 2)

                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        return img, bboxs

def finger_position(coor_list_left, coor_list_right):
    finger_indexs = [4, 8, 12, 16, 20]
    five_fingers_left, five_fingers_right = np.zeros((5, 2)), np.zeros((5, 2))
    for i in range(len(finger_indexs)):
        five_fingers_left[i, :] = coor_list_left[finger_indexs[i]]
        five_fingers_right[i, :] = coor_list_right[finger_indexs[i]]
    return five_fingers_left, five_fingers_right

def thumb_angle(coor_list_right):
    finger_indexs = [4, 3, 2]
    thumb_right = np.zeros((4, 2))
    for i in range(len(finger_indexs)):
        thumb_right[i, :] = coor_list_right[finger_indexs[i]]
    L1 = thumb_right[0, :] - thumb_right[1, :]
    L2 = thumb_right[2, :] - thumb_right[1, :]
    mod_1, mod_2 = euclidean(L1), euclidean(L2)
    cos_angle = L1.dot(L2) / (mod_1 * mod_2)
    angle = np.arccos(cos_angle) / np.pi * 180
    return angle

def compute_dx_dy(left_old, left_new):
    d_distance_left = left_new - left_old
    return d_distance_left

def euclidean(x):
    return np.sqrt(np.sum(np.square(x)))