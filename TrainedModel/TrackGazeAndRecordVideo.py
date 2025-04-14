import dlib
import cv2
import numpy as np
import pygetwindow as gw
from selenium import webdriver
import os
from pynput import keyboard
import pyautogui
import tkinter as tk
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error

# initial global variables for smoothing
gaze_x_positions = []
gaze_y_positions = []
calibration_required = False
data = []
prev_circle_id = None
current_point = 0
model_trained = False
screen_width, screen_height = pyautogui.size()


def open_website(url):
    # open the website using selenium
    options = webdriver.ChromeOptions()
    options.add_experimental_option(name="detach", value=True)  # to keep the web browser tab alive after program end
    driver = webdriver.Chrome(options=options)  # driver to control browser actions
    driver.maximize_window()  # maximize the browser window
    driver.get(url)  # open the url


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


def move_cursor(gaze_ratio_hor, gaze_ratio_ver, model, x_scaler, y_scaler):
    screen_width, screens_height = pyautogui.size()  # get screen size
    cursor_x = int(gaze_ratio_hor * screen_width / 5)  # fix x position
    cursor_y = int(gaze_ratio_ver * screens_height / 5)  # fix y position
    scaled_x_y = x_scaler.transform(np.array([[cursor_x, cursor_y]]))  # scale the inputs
    pred_x, pred_y = y_scaler.inverse_transform(model.predict(scaled_x_y))[0]  # get model predict x, y
    # smooth cursor position
    # store cursor in global variable
    global gaze_x_positions, gaze_y_positions
    gaze_x_positions.append(pred_x)
    gaze_y_positions.append(pred_y)

    # Weighted smoothing: recent data points have higher weights
    window_size = 5
    weights = np.linspace(start=1, stop=2, num=window_size) / np.sum(np.linspace(start=1, stop=2, num=window_size))

    smoothed_x = smooth_data(gaze_x_positions, window_size, weights)  # smooth the cursor position values
    smoothed_y = smooth_data(gaze_y_positions, window_size, weights)  # smooth the cursor position values
    try:
        pyautogui.moveTo(smoothed_x, smoothed_y)  # move the cursor
    except:
        print("cannot move to corner of the screen")
    return smoothed_x, smoothed_y


# Initialize control variables
cursor_control_enabled = True
stop_program = False


# function to listen key presses
def on_press(key):
    global cursor_control_enabled, stop_program
    try:
        if key.char == '2':
            cursor_control_enabled = not cursor_control_enabled
            print(f"{key.char} is pressed")
        elif key.char == 'q':
            stop_program = True
            print(f"{key.char} is pressed")
            return False  # stop listener
    except AttributeError:
        pass


def stop_calib(root):
    root.quit()
    root.destroy()


# start a separate thread for listener
listener = keyboard.Listener(on_press=on_press)
listener.start()


def gaze_calibrate(cap, detector, predictor):
    # cap = cv2.VideoCapture(1)
    # detector = dlib.get_frontal_face_detector()  # object to detect the face
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # predictor_path = os.path.join(current_dir, "../requiredfiles", "shape_predictor_68_face_landmarks.dat")
    # predictor = dlib.shape_predictor(predictor_path)
    # generate video frame
    ret, frame = cap.read()  # read the video frame
    if not ret:  # break if frame reading is unsuccessful
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to frame to gray color
    faces = detector(gray)
    try:
        for face in faces:
            landmarks = predictor(gray, face)  # detect the landmark
            gaze_ratio_left_hor, gaze_ratio_left_ver = get_gaze_ratio(eye_points=[36, 37, 38, 39, 40, 41],
                                                                      facial_landmarks=landmarks,
                                                                      gray_frame=gray)  # get the left eye gaze ratio
            gaze_ratio_right_hor, gaze_ratio_right_ver = get_gaze_ratio(eye_points=[42, 43, 44, 45, 46, 47],
                                                                        facial_landmarks=landmarks,
                                                                        gray_frame=gray)  # get the right eye gaze ratio
            gaze_ratio_hor = (gaze_ratio_left_hor + gaze_ratio_right_hor) / 2  # horizontal gaze ratio
            gaze_ratio_ver = (gaze_ratio_right_ver + gaze_ratio_right_ver) / 2
            screen_width, screens_height = pyautogui.size()  # get screen size
            cursor_x = int(gaze_ratio_hor * screen_width / 5)  # fix x position
            cursor_y = int(gaze_ratio_ver * screens_height / 5)  # fix x position
            return cursor_x, cursor_y
    except:
        print("Face not detected")
    return None


def calibration_display(n_calibration_points, root, canvas, cap, detector, predictor):
    global prev_circle_id
    global current_point
    global calibration_required
    print(
        "Calibration starting, follow the dot with your eyes. Press 'stop' anytime to terminate calibration and "
        "continue")
    if current_point >= n_calibration_points:  # check calibration point
        print("calibration finished")
        # storing current data in a file
        with open("./TrainedModel/gaze_data_whole.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(data)
        # adding current data to the dataset
        with open("./TrainedModel/gaze_data.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["screen_x", "screen_y", "x", "y"])
            writer.writerows(data)
        print("Destroying loop through the first if block")
        root.quit()
        root.destroy()
        calibration_required = False  # end calibration
        return False

    if prev_circle_id is not None:
        canvas.delete(prev_circle_id)  # delete previous gaze point
    x = np.random.randint(0, screen_width)  # generates random x value
    y = np.random.randint(0, screen_height)  # generates a random y value
    try:
        pyautogui.moveTo(x, y)  # move the cursor
        prev_circle_id = canvas.create_oval(x - 5, y - 5, x + 10, y + 10, fill='red', outline='')
        root.update()
    except:
        print("cannot move to corner of screen")
    # append gaze data
    try:
        screen_x, screen_y = gaze_calibrate(cap, detector, predictor)
        data.append([screen_x, screen_y, x, y])
        print("calibration attempt: ", len(data))
    except:
        print("gaze not detected")
        n_calibration_points += 1
    if current_point < n_calibration_points:
        root.after(2000, lambda: calibration_display(n_calibration_points, root, canvas, cap, detector, predictor))
    else:
        print("destroying loop through the second else")
        root.quit()  # Stop the Tkinter loop
        root.destroy()
    current_point += 1  # increment the calibration point


# neural network to predict gaze to screen
def nn_model():
    global model_trained
    df = pd.read_csv('./TrainedModel/gaze_data_whole.csv')  # read dataset
    # split input and target values
    x = df.iloc[:, :2].values
    y = df.iloc[:, -2:].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=1)  # split values to train and test
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(x_train)  # scale the inputs
    X_test_scaled = scaler_x.transform(x_test)  # scale the inputs

    # Scale Y
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)  # scale the inputs
    # define the model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(16, activation='relu'),
        Dense(y_train.shape[1])
    ])
    model.compile(optimizer='Adam', loss='mse', metrics=['mae'])  # compile the model
    early_stop = EarlyStopping(monitor='val_loss', patience=30,
                               restore_best_weights=True)  # early stopping to terminate training
    # train the model
    history = model.fit(X_train_scaled, y_train_scaled,
                        epochs=500, batch_size=8,
                        validation_data=(X_test_scaled, y_test_scaled),
                        callbacks=[early_stop],
                        verbose=1)
    # checking the accuracy
    nn_pred_scaled = model.predict(X_test_scaled)
    nn_preds = scaler_y.inverse_transform(nn_pred_scaled)
    r2_nn = r2_score(y_test, nn_preds)  # r2 score for accuracy
    mse = mean_squared_error(y_test, nn_preds)  # mse for accuracy
    print(f"Model training complete. The final r2 score and MSE are {r2_nn}, {mse}")
    model_trained = True
    trained_model = model
    return trained_model, scaler_x, scaler_y


def face_detect_and_record_video(url, video_output, model, x_scaler, y_scaler, webcam):
    global calibration_required, frame_count
    # start calibration when required
    cap = cv2.VideoCapture(webcam)
    detector = dlib.get_frontal_face_detector()  # object to detect the face
    current_dir = os.path.dirname(os.path.abspath(__file__))
    predictor_path = os.path.join(current_dir, "../requiredfiles", "shape_predictor_68_face_landmarks.dat")
    predictor = dlib.shape_predictor(predictor_path)  # face predictor
    # input_val = input("Do you want to calibrate gaze control? Yes/No: ").strip().lower()
    # calibration_required = input_val == "yes"
    try:
        open_website(url)  # open url in browser
    except:
        print("Cannot open URL. Please check your internet connection and relaunch tracking")
    window = gw.getWindowsWithTitle("Chrome")[-1]  # find browser window
    left, top, right, bottom = window.left, window.top, window.right, window.bottom  # get bounding box of window
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # define codec
    out = cv2.VideoWriter(video_output, fourcc, 5.0, (right - left, bottom - top))  # video capture object
    cursor_positions = []  # to store cursor position of each frame
    while True:
        if stop_program:
            break
        img = pyautogui.screenshot(region=(left, top, right - left, bottom - top))  # get screenshot of browser window
        vid_frame = np.array(img)  # generate video frame
        ret, frame = cap.read()  # read the video frame
        if not ret:  # break if frame reading is unsuccessful
            break
        prev_cursor_x, prev_cursor_y = pyautogui.position()
        vid_frame = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to frame to gray color
        faces = detector(gray)
        # cv2.line(img=vid_frame, pt1=(prev_cursor_x, prev_cursor_y), pt2=(cursor_x, cursor_y), color=(0, 255),
        # thickness=5)
        try:
            for face in faces:
                x, x1, y, y1 = face.left(), face.right(), face.top(), face.bottom()  # get the face coordinates
                cv2.rectangle(img=frame, pt1=(x, y), pt2=(x1, y1), color=(0, 255, 0),
                              thickness=2)  # draw a rectangle over the face
                landmarks = predictor(gray, face)  # detect the landmark
                gaze_ratio_left_hor, gaze_ratio_left_ver = get_gaze_ratio(eye_points=[36, 37, 38, 39, 40, 41],
                                                                          facial_landmarks=landmarks,
                                                                          gray_frame=gray)  # get the left eye gaze ratio
                gaze_ratio_right_hor, gaze_ratio_right_ver = get_gaze_ratio(eye_points=[42, 43, 44, 45, 46, 47],
                                                                            facial_landmarks=landmarks,
                                                                            gray_frame=gray)  # get the right eye gaze ratio
                gaze_ratio_hor = (gaze_ratio_left_hor + gaze_ratio_right_hor) / 2  # horizontal gaze ratio
                gaze_ratio_ver = (gaze_ratio_right_ver + gaze_ratio_right_ver) / 2  # vertical gaze ratio
                if cursor_control_enabled:
                    move_cursor(gaze_ratio_hor, gaze_ratio_ver, model, x_scaler, y_scaler)
                cursor_x, cursor_y = pyautogui.position()
                cv2.circle(img=vid_frame, center=(cursor_x, cursor_y), radius=10, color=(200, 50, 200),
                           thickness=1)  # flourescent yellow circle
                cv2.circle(img=vid_frame, center=(cursor_x, cursor_y), radius=15, color=(200, 50, 200),
                           thickness=2)  # flourescent yellow border
                cursor_positions.append([(prev_cursor_x, prev_cursor_y), (cursor_x, cursor_y)])
        except:
            print("Face not detected")
        # create an overlay image
        overlay = vid_frame.copy()

        # iterate over cursor positions to draw a line for track cursor movement
        for position in cursor_positions:
            cv2.line(img=overlay, pt1=position[0], pt2=position[1], color=(200, 50, 200),
                     thickness=10)
            cv2.addWeighted(src1=overlay, alpha=0.300, src2=vid_frame, beta=0.700, gamma=0, dst=vid_frame)
        out.write(vid_frame)  # write the frame to the video file
        # cv2.imshow("Frame", frame)  # display the frame
        # check key press to quit
        if stop_program:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    face_detect_and_record_video("https://example.com", "output.avi")
