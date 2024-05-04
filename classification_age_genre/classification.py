import cv2
from picamera2 import Picamera2
import numpy as np

#camera = "/base/soc/i2c0mux/i2c@1/imx219@10"
#pipeline = "libcamerasrc camera-name=%s ! video/x-raw,width=640,height=480,framerate=10/1,format=RGBx ! videoconvert ! videoscale ! video/x-raw,width=640,height=480,format=BGR ! appsink" % (camera)
 
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes
 
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
 
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
 
faceNet = cv2.dnn.readNet(faceModel, faceProto)  
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

""" picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (800, 800)}))
picam2.start() """

picam2 = Picamera2()
picam2.preview_configuration.main.size = (1000,1000)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()
 
 
#video = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER) #ancienne méthode 
frame = picam2.capture_array() 
#print(frame)
padding = 20

 
desired_window_width = 800
ratio = desired_window_width / frame.shape[1]
height = int(frame.shape[0] * ratio)

cv2.startWindowThread()
cv2.namedWindow("Detecting age and gender")
image_noire = np.zeros((1000, 1000, 3), dtype=np.uint8)
cv2.imshow("Detecting age and gender", image_noire)
 
while True:
    frame = picam2.capture_array()
    

    frame = cv2.resize(frame, (desired_window_width, height))
 
    resultImg, faceBoxes = highlightFace(faceNet, frame) #probleme ici
    
    if not faceBoxes:  #pas de visage detecte
        cv2.imshow("Detecting age and gender", resultImg) #affichage de l'image telle quelle
        
    else:
        # ne rentre que si visage detecte donc n'affiche rien sinon
        for faceBox in faceBoxes: 
            face = frame[
                max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)
            ]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
    
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
    
            text = "{}:{}".format(gender, age)
            cv2.putText(resultImg, text, (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Detecting age and gender", resultImg)
 
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
 
cv2.destroyAllWindows()
