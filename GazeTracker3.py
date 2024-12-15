import dlib
import cv2
import pyautogui
import numpy as np
from math import hypot

# initial global variables for smoothing
gaze_x_positions = []
gaze_y_positions = []
calibration_data = {"left": None, "center": None, "right": None, "top": None,
                    "bottom": None}  # to store calibration points


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def blink_ratio(eye_points, facial_landmarks):
    left_point = (
        facial_landmarks.part(eye_points[0]).x,
        facial_landmarks.part(eye_points[0]).y)  # marking left point around pupil
    right_point = (
        facial_landmarks.part(eye_points[3]).x,
        facial_landmarks.part(eye_points[3]).y)  # marking right point around pupil
    center_top = midpoint(facial_landmarks.part(eye_points[1]),
                          facial_landmarks.part(eye_points[2]))  # marking the center top around pupil
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]),
                             facial_landmarks.part(eye_points[4]))  # marking the center bottom around pupil
    hor_line_length = hypot((left_point[0] - right_point[0]),
                            left_point[1] - right_point[1])  # to measure vertical distance of eye
    ver_line_length = hypot((center_top[0] - center_bottom[0]),
                            (center_top[1] - center_bottom[1]))  # to measure horizontal distance of eye
    return hor_line_length / ver_line_length


def get_gaze_ratio(eye_points, facial_landmarks, gray_frame):
    eye_region = np.array(
        [(facial_landmarks.part(eye_points[i]).x, facial_landmarks.part(eye_points[i]).y) for i in range(6)],
        dtype=np.int32)  # convert the eye landmarks into array
    height, width = gray_frame.shape  # get the dimensions of the gray frame
    mask = np.zeros((height, width), np.uint8)  # create a mask
    cv2.polylines(img=mask, pts=[eye_region], isClosed=True, color=(255, 255, 255), thickness=2)  # display lines
    cv2.fillPoly(img=mask, pts=[eye_region], color=(255, 255, 255))
    eye = cv2.bitwise_and(src1=gray_frame, src2=gray_frame, mask=mask)

    min_x, max_x = np.min(eye_region[:, 0]), np.max(eye_region[:, 0])  # to get x-coordinate of eye
    min_y, max_y = np.min(eye_region[:, 1]), np.max(eye_region[:, 1])  # to get y-coordinate of eye
    gray_eye = eye[min_y:max_y, min_x:max_x]  # to separate the eye region from the frame

    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)
    height, width = threshold_eye.shape  # get the shape of threshold eye
    left_side_white = cv2.countNonZero(threshold_eye[:, :width // 2])  # count the left side white pixels of the eye
    right_side_white = cv2.countNonZero(threshold_eye[:, width // 2:])  # count the right side white pixels of the eye
    top_side_white = cv2.countNonZero(threshold_eye[: height // 2, :])  # count the right side white pixels of the eye
    bottom_side_white = cv2.countNonZero(threshold_eye[height // 2:, :])  # count the right side white pixels of the eye

    if right_side_white == 0:
        gaze_ratio_hor = 1
    else:
        gaze_ratio_hor = left_side_white / right_side_white  # gaze ratio to determine which side the user is looking
        # horizontally

    if bottom_side_white == 0:  # gaze ratio to determine which side the user is looking
        # vertically
        gaze_ratio_ver = 1
    else:
        gaze_ratio_ver = top_side_white / bottom_side_white

    return gaze_ratio_hor, gaze_ratio_ver


# function to smoothen the gaze points
def smooth_data(data, window_size=5, weights=None):
    if len(data) < window_size:
        return data[-1]
    if weights is None:
        weights = np.ones(window_size) / window_size
    return np.dot(data[-window_size:], weights)


def move_cursor(gaze_ratio_hor, gaze_ratio_ver):
    screen_width, screens_height = pyautogui.size()  # get screen size
    cursor_x = int(gaze_ratio_hor * screen_width / 5)  # fix x position
    cursor_y = int(gaze_ratio_ver * screens_height / 5)  # fix y position

    # smooth cursor position
    global gaze_x_positions, gaze_y_positions
    gaze_x_positions.append(cursor_x)  # store cursor in global variable
    gaze_y_positions.append(cursor_y)

    # Weighted smoothing: recent data points have higher weights
    window_size = 5
    weights = np.linspace(start=1, stop=2, num=window_size) / np.sum(np.linspace(start=1, stop=2, num=window_size))

    smoothed_x = smooth_data(gaze_x_positions, window_size, weights)  # smooth the cursor position values
    smoothed_y = smooth_data(gaze_y_positions, window_size, weights)  # smooth the cursor position values
    pyautogui.moveTo(smoothed_x, smoothed_y)  # move the cursor


def calibrate():
    cap = cv2.VideoCapture(0)  # capture vide
    detector = dlib.get_frontal_face_detector()  # object to detect face
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # predict face shape using landmarks
    # initial calibration points
    calibration_points = {"left": (50, pyautogui.size().height // 2),
                          "center": (pyautogui.size().width // 2, pyautogui.size().height // 2),
                          "right": (pyautogui.size().width - 50, pyautogui.size().height // 2),
                          "top": (pyautogui.size().width // 2, 50),
                          "bottom": (pyautogui.size().width // 2, pyautogui.size().height - 50)
                          }

    for point, (x, y) in calibration_points.items():
        print(f"look at the {point} point on the screen for calibration")
        pyautogui.moveTo(x, y)  # moves the cursor based on calibration points
        cv2.waitKey(3000)  # 3 seconds # time for the user to look at the data
        ret, frame = cap.read()  # captures the video frame
        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)  # converts the frame to gray scale
        faces = detector(gray)  # face detector
        if len(faces) > 0:
            face = faces[0]
            landmarks = predictor(gray, face)  # extract facial landmarks
            gaze_ratio_hor, gaze_ratio_ver = get_gaze_ratio(eye_points=[36, 37, 38, 39, 40, 41],
                                                            facial_landmarks=landmarks,
                                                            gray_frame=gray)  # calculate horizontal and vertical
            # gaze ratio
            calibration_data[point] = (gaze_ratio_hor, gaze_ratio_ver)  # stores the gaze ratio in current
            # calibration point
            print(f"Calibration data for {point}: {calibration_data[point]}")
    cap.release()
    print("Calibration complete:", calibration_data)


def face_detect():
    calibrate()
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()  # object to detect the face
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    while True:
        ret, frame = cap.read()  # read the video frame
        if not ret:  # break if frame reading is unsuccessful
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to frame to gray color
        faces = detector(gray)
        for face in faces:
            x, x1, y, y1 = face.left(), face.right(), face.top(), face.bottom()  # get the face coordinates
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x1, y1), color=(0, 255, 0),
                          thickness=2)  # draw a rectangle over the face
            landmarks = predictor(gray, face)  # detect the landmark
            # left_eye_ratio = blink_ratio(eye_points=[36, 37, 38, 39, 40, 41], facial_landmarks=landmarks,
            #                              frame=frame)  # get the vertical ratio of left eye
            # right_eye_ratio = blink_ratio(eye_points=[42, 43, 44, 45, 46, 47], facial_landmarks=landmarks,
            #                               frame=frame)  # get the vertical ratio of right eye
            # blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2  # averages the blink ratio
            # if blinking_ratio > 4.7:  # detect the blink
            #     print(blinking_ratio)
            #     cv2.putText(img=frame, text="Blinking", org=(50, 150), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=4,
            #                 color=(255, 0, 0))  # displays the blinking text

            gaze_ratio_left_hor, gaze_ratio_left_ver = get_gaze_ratio(eye_points=[36, 37, 38, 39, 40, 41],
                                                                      facial_landmarks=landmarks,
                                                                      gray_frame=gray)  # get the left eye gaze ratio
            gaze_ratio_right_hor, gaze_ratio_right_ver = get_gaze_ratio(eye_points=[42, 43, 44, 45, 46, 47],
                                                                        facial_landmarks=landmarks,
                                                                        gray_frame=gray)  # get the right eye gaze ratio
            gaze_ratio_hor = (gaze_ratio_left_hor + gaze_ratio_right_hor) / 2  # horizontal gaze ratio
            gaze_ratio_ver = (gaze_ratio_right_ver + gaze_ratio_right_ver) / 2  # vertical gaze ratio

            if calibration_data["center"]:
                center_hor, center_ver = calibration_data["center"]
                print(f"Center calibration data: {center_hor}, {center_ver}")
                gaze_ratio_hor -= center_hor - 1
                gaze_ratio_ver -= center_ver - 1
            else:
                print("Center calibration data not found")

            move_cursor(gaze_ratio_hor, gaze_ratio_ver)

        cv2.imshow("Frame", frame)  # display the frame
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
