import copy

from flask import Flask, render_template, request, send_file
import os
import cv2
import numpy as np
from MakeVideo import make_video
from GazeTracker import eye_detection
#from GazeTracker2 import face_detect
from GazeTracker3 import face_detect
#print(cv2.data.haarcascades + 'haarcascade_eye.xml')

detect_eye = face_detect()


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
HEATMAP_FOLDER = 'heatmaps'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
from progress.bar import Bar


def create_heatmap(video_path):
    # Initialize video capture
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    # Initialize background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2()
    heatmap_intensity = np.zeros((480, 640), np.float32)  # Assuming video has a resolution of 640x480
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bar = Bar('Processing Frames', max=length)  #create bar

    try:
        if not os.path.exists('frames'):
            os.makedirs('frames')
    except OSError:
        print("Could not create the directory")

    first_iteration_indicator = 1
    for i in range(0, length):
        ret, frame = cap.read()

        if first_iteration_indicator==1:
            first_frame = copy.deepcopy(frame)
            height, width = frame.shape[:2]
            accum_images = np.zeros((height, width), np.uint8)
            first_iteration_indicator = 0

        else:
            filter = backSub.apply(frame) #remove the background
            cv2.imwrite('./frame.jpg', frame)
            cv2.imwrite('./diff-bkgnd-frame.jpg', filter)

            threshold = 2
            max_value = 2
            ret, th1 = cv2.threshold(filter, threshold, max_value, cv2.THRESH_BINARY)

            #gamma correction to make the red tracing more bright
            gamma = 2.0  # Gamma value
            look_up_table = np.array([((j / 255.0) ** gamma) * 255 for j in np.arange(0, 256)]).astype("uint8")

            #add accumulated image
            accum_images = cv2.add(accum_images, th1)
            cv2.imwrite('./mask.jpg', accum_images)

            color_image_video = cv2.applyColorMap(accum_images, cv2.COLORMAP_SUMMER)
            #color_image_video = cv2.LUT(color_image_video, look_up_table)
            video_frame = cv2.addWeighted(frame, 0.3, color_image_video, 1.0, 0)

            color_image = cv2.applyColorMap(accum_images, cv2.COLORMAP_HOT)
            #color_image = cv2.LUT(color_image, look_up_table)
            result_overlay = cv2.addWeighted(first_frame, 0.1, color_image, 5, 50)

            #detect cursor
            # lower_bound = (200, 200, 200)
            # upper_bound = (255, 255, 255)
            # cv2.inRange(result_overlay, lower_bound, upper_bound)

            name = "./frames/frame%d.jpg" % i
            cv2.imwrite(name, result_overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        bar.next()
    bar.finish()

    make_video('./frames/', 'output.avi')
    cv2.imwrite('diff-overlay.jpg', result_overlay)

    cap.release()
    cv2.destroyAllWindows()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for consistency
        frame = cv2.resize(frame, (640, 480))
        fgMask = backSub.apply(frame)

        # Update heatmap intensity
        heatmap_intensity += fgMask

    cap.release()
    # Normalize heatmap
    heatmap_intensity = cv2.normalize(heatmap_intensity, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(heatmap_intensity.astype(np.uint8), cv2.COLORMAP_JET)

    # Save heatmap
    heatmap_path = os.path.join(HEATMAP_FOLDER, 'heatmap.png')
    cv2.imwrite(heatmap_path, heatmap_colored)
    heatmap_path = os.path.join('./frames/', 'output.avi')

    return heatmap_path


@app.route('/')
def index():
    return render_template('uploadnew.html')


@app.route('/uploadnew', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return "No file part"

    file = request.files['video']
    print(file)
    if file.filename == '':
        return "No selected file"

    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    print(f"Saving file to {video_path}")
    file.save(video_path)

    heatmap_path = create_heatmap(video_path)


    return send_file(heatmap_path, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
