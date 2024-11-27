import dlib
import cv2
import pyautogui


def eye_detection():
    detector = dlib.get_frontal_face_detector()  # initializing the face detector
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml')  # classifies objects in a dataset in this case eye
    if eye_cascade.empty():
        print("Error: Failed to load haarcascade_eye.xml")
    else:
        print("Haarcascade for eye detection loaded successfully!")

    cap = cv2.VideoCapture(0)  # read the video from webcam

    screen_width, screen_height = pyautogui.size() # get the primary screen width and height

    while True:
        ret, frame = cap.read()  # reads the frame in video. ret indicates success or failure of the image capture
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convers the color space of the frame to gray

        faces = detector(gray) #detect faces

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height() #detect the face dimensions
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2) #form a traingle around the face with blue borders

            roi_gray = gray[y:y+h, x:x+w] #isolate the face from the image
            roi_color = frame[y:y + h, x:x + w] #isolate the region of interest
            eyes = eye_cascade.detectMultiScale(roi_gray) #detect eyes from the image returning list of identified eyes

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2) #adding rectangular border to detect the eyes
                eye_frame = roi_color[ey: ey + eh, ex:ex + ew]
                pupil_center = detect_pupil(eye_frame)
                if pupil_center:
                    cv2.circle(roi_color, (ex + pupil_center[0], ey + pupil_center[1]), 5, (0, 0, 255), -1)
                    screen_x = int((ex + pupil_center[0]) * screen_width / frame.shape[1]) # maps the x position of the eye in the frame to the screenwidth
                    screen_y = int((ey + pupil_center[1]) * screen_height / frame.shape[0]) # maps the y position of the eye in the frame ot the screenheight
                    pyautogui.moveTo(screen_x, screen_y)

        cv2.imshow('frame', frame) #displays the video

        if cv2.waitKey(1) & 0xFF==ord('q'): ## exits the video when q is pressed for 1 second
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_pupil(eye_frame):
        gray_eye = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY) #converts color space to gray scale
        _, threshold =  cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV) #applies threshold to convert it into binary image
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #apply contours to detect eye effectively

        contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            return (x + w//2, y + h//2)
        return None

