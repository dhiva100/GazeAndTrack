# GazeAndTrack
Control a website using gaze and generate heatmap video of the cursor movements
The gaol is to open the website of our interest and records the user's website interactions as video. The video need to be then processed to generate a heatmap video in which the user cursor movements (basically the areas in the website in which user gaze fell) is highlighted.
**Steps to Implement the Project:**
	a. Open the Website of Interest:
		○ Automated Browser Control: Use tools like Selenium or Puppeteer to automate the opening and interaction with the website.
	b. Record User Interactions:
		○ Gaze Tracking: Implement a gaze tracker. Tools like Tobii or OpenGazer can help.
		○ Record Screen and Interactions: Use libraries like OpenCV, PyAutoGUI, or ffmpeg to record the screen and cursor movements.
	c. Process Video to Generate Heatmap:
		○ Extract Cursor Movements: Analyze the recorded video to extract cursor positions. OpenCV can help in processing video frames.
     Generate Heatmap: Use the extracted cursor position
