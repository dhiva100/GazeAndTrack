import os
import threading

import cv2
import numpy as np
from flask import Flask, render_template, request, send_file, jsonify
from TrainedModel.TrackGazeAndRecordVideo import face_detect_and_record_video
from TrainedModel.TrackGazeAndRecordVideo import calibration_display, stop_calib, nn_model
import copy
import dlib
import cv2
import tkinter as tk
import pyautogui
import TrainedModel.TrackGazeAndRecordVideo as tg

app = Flask(__name__)
INPUT_FOLDER = 'VideoInput'
OUTPUT_FOLDER = 'HeatmapOutput'
HEATMAP_FOLDER = 'heatmaps'
current_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(INPUT_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
input_video_path = os.path.join(current_dir, INPUT_FOLDER, "GazeRecordedVideo.avi")
output_video_path = os.path.join(current_dir, OUTPUT_FOLDER, "GazeHeatMap.avi")
parameters = {}
model = None
x_scaler = None
y_scaler = None
webcam = 0


# function to create video from frames
def make_video(frame_folder, out_video_path):
    frame_files = [f for f in os.listdir(frame_folder) if f.endswith('.jpg')]
    frame_files.sort()
    # set the first frame
    first_frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
    height, width, _ = first_frame.shape

    # video writer object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    # add the frames to video and delete
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frame_folder, frame_file))
        out.write(frame)
        os.unlink(os.path.join(frame_folder, frame_file))

    out.release()
    print(f"Video saved as {out_video_path}")


def create_heatmap(video_path):
    cap = cv2.VideoCapture(video_path)  # video capture object
    backSub_filter = cv2.createBackgroundSubtractorMOG2()  # background separator object
    heatmap_intensity = np.zeros((480, 640), np.float32)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # get the frame count

    #  create the frames directory if it does not exist
    if not os.path.exists('frames'):
        os.makedirs('frames')

    first_iteration_indicator = 1
    # loop through frames
    for i in range(0, length):
        ret, frame = cap.read()
        # store the first frame to set as heatmap background
        if first_iteration_indicator == 1:
            first_frame = copy.deepcopy(frame)
            height, width = frame.shape[:2]
            accum_images = np.zeros((height, width), np.uint8)
            first_iteration_indicator = 0
        else:
            bg_filter = backSub_filter.apply(frame)  # background fileter to identify moving object
            threshold = 2
            max_value = 2
            ret, th1 = cv2.threshold(bg_filter, threshold, max_value, cv2.THRESH_BINARY)  # apply binary threshold

            accum_images = cv2.add(accum_images, th1)
            color_image_video = cv2.applyColorMap(accum_images, cv2.COLORMAP_SUMMER)  # apply heatmap color code
            video_frame = cv2.addWeighted(frame, 0.3, color_image_video, 1.0, 0)  # store the processed frames

            color_image = cv2.applyColorMap(accum_images, cv2.COLORMAP_HOT)  # apply heatmap color code
            result_overlay = cv2.addWeighted(first_frame, 0.1, color_image, 5, 50)  # store the processed frames

            frame_name = f"./frames/frame{i}.jpg"
            cv2.imwrite(frame_name, result_overlay)  # store the first frames as image

    make_video('./frames/', output_video_path)  # call the object to create video

    cap.release()
    cv2.destroyAllWindows()

    # Generate the heatmap
    heatmap_intensity = cv2.normalize(heatmap_intensity, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(heatmap_intensity.astype(np.uint8), cv2.COLORMAP_JET)

    heatmap_path = os.path.join(HEATMAP_FOLDER, 'heatmap.png')
    cv2.imwrite(heatmap_path, heatmap_colored)

    return video_path  # Return the path to the video file


# record_gaze = input("Do you want to record gaze? Yes/No: ")
# if record_gaze == "Yes":
#     print("Gaze control is starting now. Always use key 't' to toggle gaze control and 'q' to terminate gaze control "
#           "and continue with generating heatmap video\n")
#     url = str(input("Enter the website URL: "))
#     face_detect_and_record_video(url, input_video_path)
#     create_heatmap(input_video_path)
# else:
#     print("Please upload the input video to generate heatmap")
def calibration_thread(n_points, cam):
    global parameters
    cap = cv2.VideoCapture(cam)  # open webcam
    detector = dlib.get_frontal_face_detector()  # face detector
    predictor = dlib.shape_predictor("requiredfiles/shape_predictor_68_face_landmarks.dat")  # path to model

    screen_width, screen_height = pyautogui.size()
    root = tk.Tk()
    root.attributes('-fullscreen', True)
    root.attributes('-topmost', True)
    root.attributes('-alpha', 0.3)
    root.config(bg='black')
    root.overrideredirect(True)

    canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg='black', highlightthickness=0)
    root.bind('s', lambda event: stop_calib(root))
    canvas.pack()

    # n_points = parameters["setup"]["n_calib_points"]

    # Start calibration
    def start_calibration():
        calibration_display(n_points, root, canvas, cap, detector, predictor)

    root.after(100, start_calibration)
    root.mainloop()
    cap.release()


@app.route('/')
def index():
    return render_template('uploadnew.html')


@app.route('/uploadnew', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return "No file part"

    file = request.files['video']
    if file.filename == '':
        return "No selected file"

    file.save(input_video_path)

    video_file_path = create_heatmap(input_video_path)

    # Return the video file for download
    return send_file(video_file_path, as_attachment=True)


@app.route('/setup', methods=['POST'])
def setup():
    data = request.get_json()
    global parameters, webcam
    tracking = data.get('tracking', False)
    tg.calibration_required = data.get('calibration', False)
    n_calib_points = int(data.get('n_calib_points', 0))
    target_url = data.get('url', "")
    try:
        webcam = int(data.get('webcam', 0))
    except ValueError:
        return jsonify({"error": "Webcam value must be an integer."}), 400

    if webcam not in [0, 1]:
        return jsonify({"error": "Invalid webcam index. Only 0 (inbuilt) or 1 (external) allowed."}), 400
    parameters["setup"] = {"tracking": tracking,
                           "calibration": tg.calibration_required,
                           "n_calib_points": n_calib_points,
                           "target_url": target_url,
                           "webcam": webcam}
    print("setup received:", parameters["setup"])
    print("Using", "External camera" if webcam == 1 else "Inbuilt camera")
    return jsonify({"status": "config received"})


@app.route('/calibrate', methods=["POST"])
def calibrate():
    n_points = parameters["setup"]["n_calib_points"]
    cam = parameters["setup"]["webcam"]
    threading.Thread(target=calibration_thread, args=(n_points, cam)).start()
    parameters["setup"]["calibration"] = False
    return jsonify({"status": "Calibration complete"})


@app.route('/train', methods=["POST"])
def train():
    global model, x_scaler, y_scaler
    model, x_scaler, y_scaler = nn_model()
    return jsonify({"status": "Model Trained"})


@app.route('/start-tracking', methods=['POST'])
def start_tracking():
    global model, x_scaler, y_scaler
    cam = parameters["setup"]["webcam"]

    def tracking_thread():
        face_detect_and_record_video(parameters["setup"]["target_url"], "./VideoInput/recorded_video.avi", model=model,
                                     x_scaler=x_scaler, y_scaler=y_scaler, webcam=cam)

    threading.Thread(target=tracking_thread).start()
    parameters["setup"]["tracking"] = False
    return jsonify({"status": "tracking started"})


@app.route('/stop', methods=['POST'])
def stop_tracking():
    tg.stop_program = True
    return jsonify({"status": "Tracking stopped"})


@app.route('/toggle-cursor', methods=['POST'])
def toggle_cursor():
    tg.cursor_control_enabled = not tg.cursor_control_enabled
    return jsonify({
        "status": "Cursor control toggled",
        "enabled": tg.cursor_control_enabled
    })


@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "model_trained": model is not None,
        "tracking_enabled": tg.cursor_control_enabled,
        "calibration_mode": tg.calibration_required
    })


@app.route('/reset', methods=['POST'])
def reset():
    global stop_progam, model, x_scaler, y_scaler
    model = None
    x_scaler = None
    y_scaler = None
    tg.calibration_required = True
    return jsonify({"status": "Reset successful"})


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
