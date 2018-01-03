# Face Recognition

# Importing the libraries
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('Nariz.xml')

# Defining a function that will do the detections
def detect_and_apply(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
 


    for (x, y, w, h) in faces:
        
        smile_bool = False
        

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 23)
        if len(smile) == 0:
            smile_bool= True
        for (ex, ey, ew, eh) in smile:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    
        nose = nose_cascade.detectMultiScale(roi_gray, 1.7, 8)
        if smile_bool:
            my_l_ear = cv2.imread("dog_l_ear.png", -1)
            my_r_ear = cv2.imread("dog_r_ear.png", -1)
            my_nose = cv2.imread("dog_nose.png", -1)
            my_smile = cv2.imread("dog_tongue.png", -1)
        else:
            my_l_ear = cv2.imread("rab_l_ear.png", -1)
            my_r_ear = cv2.imread("rab_r_ear.png", -1)
            my_nose = cv2.imread("rab_nose.png", -1)
            my_smile = cv2.imread("rab_tongue.png", -1)
        my_l_ear = cv2.resize(my_l_ear, (int(w/4), int(h/4)))
        my_r_ear = cv2.resize(my_r_ear, (int(w/4), int(h/4)))
        
        for (nx, ny, nw, nh) in nose:
             cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (0, 0, 255), 2)
             '''
                 TODO: Apply nose filter
             rab_nose = cv2.resize(my_nose, (nw,nh))
             
             y1, y2 = y+ny, y+ny+nw
             x1, x2 = x+nx, x+nx+nw
        
             alpha_s = my_nose[:, :, 3] / 255.0
             alpha_l = 1.0 - alpha_s
        
             for c in range(0, 3):
                frame[y1:y2, x1:x2, c] = (alpha_s * my_nose[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])
             break
             '''

        y1, y2 = y, y+int(w/4)
        x1, x2 = x, x+int(h/4)

        alpha_s = my_l_ear[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        #APPLY EARS
        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha_s * my_l_ear[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])
        y1, y2 = y, y+int(w/4)
        x1, x2 = x+h-int(h/4), x+h
    
        alpha_s = my_r_ear[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        
        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha_s * my_r_ear[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])

    return frame

# Test the detect and apply code with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect_and_apply(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
