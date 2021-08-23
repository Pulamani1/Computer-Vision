import cv2
import mediapipe as mp
import time
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
cap = cv2.VideoCapture('run1.mp4')
pTime = 0
def rescaleImg(img,scale=0.5):
    width = int(img.shape[1]* scale)
    height = int(img.shape[0] * scale)
    dimension = (width, height)
    return cv2.resize(img,dimension, interpolation = cv2.INTER_AREA)
while True:
    success, img = cap.read()
    img_resized = rescaleImg(img)
    imgRGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img_resized, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img_resized.shape
            print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img_resized, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    cv2.imshow("Image", img_resized)
    cv2.waitKey(1)