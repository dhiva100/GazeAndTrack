import dlib
import cv2
import pyautogui
from math import hypot
import numpy as np


# function to get the midpoint of two points
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


# function to return the blink ratio of an eye
def blink_ratio(eye_points, facial_landmarks, frame):
    left_point = (
        facial_landmarks.part(eye_points[0]).x,
        facial_landmarks.part(eye_points[0]).y)  # marking left point around pupil
    right_point = (
        facial_landmarks.part(eye_points[3]).x,
        facial_landmarks.part(eye_points[3]).y)  # marking right point around pupil
    center_top = midpoint(facial_landmarks.part(eye_points[1]),
                          facial_landmarks.part(eye_points[2]))  # marking the center top around pupil
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]),
                             facial_landmarks.part(eye_points[4]))  # marking the center bottm around pupil
    # hor_line = cv2.line(img=frame, pt1=left_point, pt2=right_point, color=(0, 255, 0),
    # thickness=2)  # drawing a horizantal line in pupil
    # er_line = cv2.line(img=frame, pt1=center_top, pt2=center_bottom, color=(0, 255, 0),
    # thickness=2)  # drawing a vertical line in pupil
    hor_line_length = hypot((left_point[0] - right_point[0]),
                            left_point[1] - right_point[1])  # to measure vertical distance of eye
    ver_line_length = hypot((center_top[0] - center_bottom[0]),
                            (center_top[1] - center_bottom[1]))  # to measure horizontal distance of eye
    ratio = hor_line_length / ver_line_length
    return ratio


def get_gaze_ratio(eye_points, facial_landmarks, frame, gray_frame):
    eye_region = np.array([(facial_landmarks.part(0).x, facial_landmarks.part(0).y),
                           (facial_landmarks.part(1).x, facial_landmarks.part(1).y),
                           (facial_landmarks.part(2).x, facial_landmarks.part(2).y),
                           (facial_landmarks.part(3).x, facial_landmarks.part(3).y),
                           (facial_landmarks.part(4).x, facial_landmarks.part(4).y),
                           (facial_landmarks.part(5).x, facial_landmarks.part(5).y)],
                          dtype=np.int32)  # convert the left eye landmarks pixels asarray
    height, width, _ = frame.shape  # get the dimensions of the frame
    mask = np.zeros((height, width), np.uint8)  # create a mask
    # cv2.polylines(img=frame, pts=[left_eye_region], isClosed=True, color=(0, 0, 255), thickness=2) #draw a circle around the eyeball
    cv2.polylines(img=mask, pts=[eye_region], isClosed=True, color=(255, 255, 255), thickness=2)  # display lines
    cv2.fillPoly(img=mask, pts=[eye_region], color=(255, 255, 255))
    eye = cv2.bitwise_and(src1=gray_frame, src2=gray_frame, mask=mask)

    if eye_region[:0].size > 0:  # find the x coordinates of left eye
        print("values are used in x")
        min_x = np.min(eye_region[:0])
        max_x = np.max(eye_region[:0])
    else:
        print("none is used in x")
        min_x = None
        max_x = None

    if eye_region[:1].size > 0:  # find the y coordinates of right eye
        print("values are used in y")
        min_y = np.min(eye_region[:1])
        max_y = np.max(eye_region[:1])
    else:
        print("none is used in y")
        min_y = None
        max_y = None

    gray_eye = eye[min_y:max_y, min_x:max_x]  # to separate the eye region from the frame
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    threshold_eye = cv2.resize(src=threshold_eye, dsize=None, fx=5, fy=5)  # resize the threshold frame
    eye = cv2.resize(src=gray_eye, dsize=None, fx=5, fy=5)  # resize the eyeframe
    print("shape", threshold_eye.shape)
    height, width = threshold_eye.shape  # get the shape of threshold eye
    left_side_threshold = threshold_eye[0:height, 0:int(width / 2)]  # get the dimensions of left of the threshold eye
    left_side_white = cv2.countNonZero(left_side_threshold)  # count the white pixels

    right_side_threshold = threshold_eye[0:height,
                           int(width / 2):width]  # get the dimensions of right half of the threshold eye
    right_side_white = cv2.countNonZero(right_side_threshold)  # count the white pixels

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white

    return gaze_ratio


def face_detect():
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()  # object to detect the face
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    while True:
        _, frame = cap.read()  # read the video frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to frame to gray color
        faces = detector(gray)
        new_frame = np.zeros((500, 500, 3), np.uint8)  # to display color based on gaze direction
        for face in faces:
            x, x1, y, y1 = face.left(), face.right(), face.top(), face.bottom()  # get the face coordinates
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x1, y1), color=(0, 255, 0),
                          thickness=2)  # draw a rectange over the face
            landmarks = predictor(gray, face)  # detect the landmark

            left_eye_ratio = blink_ratio(eye_points=[36, 37, 38, 39, 40, 41], facial_landmarks=landmarks,
                                         frame=frame)  # get the vertical ratio of left eye
            right_eye_ratio = blink_ratio(eye_points=[42, 43, 44, 45, 46, 47], facial_landmarks=landmarks,
                                          frame=frame)  # get the vertical ratio of right eye
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2  # averages the blink ratio
            if blinking_ratio > 4.7:  # detect the blink
                print(blinking_ratio)
                cv2.putText(img=frame, text="Blinking", org=(50, 150), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=4,
                            color=(255, 0, 0))  # diplays the blinking text

            gaze_ratio_left_eye = get_gaze_ratio(eye_points=[36, 37, 38, 39, 40, 41], facial_landmarks=landmarks,
                                                 frame=frame, gray_frame=gray)  # get the left eye gaze ratio

            gaze_ratio_right_eye = get_gaze_ratio(eye_points=[42, 43, 44, 45, 46, 47], facial_landmarks=landmarks,
                                                  frame=frame, gray_frame=gray)  # get the right eye gaze ratio

            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

            if gaze_ratio <= 1:
                gaze_message = "right"
                new_frame[:] = (0, 0, 255)
            elif 1 < gaze_ratio < 3:
                gaze_message = "center"
                new_frame[:] = (255, 0, 0)
            else:
                new_frame[:] = (0, 255, 0)
                gaze_message = "left"

            cv2.putText(img=frame, text=str(gaze_message), org=(50, 100), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2,
                        color=(0, 0, 255), thickness=3)  # displays the number of leftsidewhite pixels
        cv2.imshow("Frame", frame)  # display the frame
        cv2.imshow("Direction", new_frame)

        key = cv2.waitKey(1)  # key to break the loop
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
