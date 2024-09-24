import cv2
video_capture = cv2.VideoCapture('1.mp4')  
if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()
ret, prev_frame = video_capture.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
   # frame = cv2.resize(frame, (600,600))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = cv2.resize(frame, (600, 600))
    frame_diff = cv2.absdiff(prev_frame_gray, frame_gray)
   
    _, thresholded = cv2.threshold(frame_diff, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('motion Detection', frame)
    prev_frame_gray = frame_gray
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
