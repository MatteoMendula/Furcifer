import sys
import cv2 
import imutils
from yoloDet import YoloTRT
import time

# use path for library and engine file
model = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")

#cap = cv2.VideoCapture("videos/testvideo.mp4")
cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
prev = 0
img_index = 0

print("Everything loaded")
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    detections, t = model.Inference(frame)
    # for obj in detections:
    #    print(obj['class'], obj['conf'], obj['box'])
    time_now = time.time()
    time_elapsed = time_now - prev
    print("FPS = {}".format(time_elapsed))
    prev = time_now
    #print("FPS: {} sec".format(1/t))
    cv2.imshow("Output", frame)
    #cv2.imwrite("./detection_camera/img{}.jpg".format(img_index), frame)
    img_index += 1
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()