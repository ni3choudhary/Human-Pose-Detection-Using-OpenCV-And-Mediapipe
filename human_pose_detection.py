import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('dance.mp4')

mppose = mp.solutions.pose
pose = mppose.Pose() 
mpdraw = mp.solutions.drawing_utils

previous_time = 0
current_time = 0

while True:
    
    _ , frame = cap.read()
    frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_LINEAR) 
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img)
    
    if results.pose_landmarks:
        mpdraw.draw_landmarks(frame, results.pose_landmarks, mppose.POSE_CONNECTIONS)
        
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    
    cv2.putText(frame, "frame rate: "+str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    cv2.imshow('Pose Detection', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
