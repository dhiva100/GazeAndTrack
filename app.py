import os
import cv2
import numpy as np
from flask import Flask, render_template, request, send_file
from TrackGazeAndRecordVideo import face_detect_and_record_video
import copy

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


record_gaze = input("Do you want to record gaze? Yes/No: ")
if record_gaze == "Yes":
    print("Gaze control is starting now. Always use key 't' to toggle gaze control and 'q' to terminate gaze control "
          "and continue with generating heatmap video\n")
    url = str(input("Enter the website URL: "))
    face_detect_and_record_video(url, input_video_path)
    create_heatmap(input_video_path)
else:
    print("Please upload the input video to generate heatmap")


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


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0")
