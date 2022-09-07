import cv2

# Our Image
img_file = 'Car Image.jpg'
video = cv2.VideoCapture('DashCam_Trim.mp4')


#Our pre-trained car and pedestrian classifier
car_tracker_file='car_detector.xml'
pedestrian_tracker_file='pedestrian_detector.xml'

# create car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

# Run Forever Until Car Stops or Crashes
while True:

    #Read the current frame
    (read_successful, frame) = video.read()

    if read_successful:
        # Must convert to grayscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscale_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscale_frame)

    # draw rectangles around the cars and pedestrians
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)    

    # Display the image with the faces spotted
    cv2.imshow('Car and Pedestrian Detector', frame)

    # Don't Autoclose (Wait here in the code and listen for a key press)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key==81 or key==113:
        break

# Release the VideoCapture object
video.release()
