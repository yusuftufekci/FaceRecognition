import cv2
import os
import numpy as np
import serial



def detectFace(test_img):
    grayImg=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)#convert color image to grayscale
    haar_cascade=cv2.CascadeClassifier('C:\\Users\\Gokturk\\Downloads\\haarcascade_frontalface_default.xml')#Load haar classifier
    detected_faces=haar_cascade.detectMultiScale(grayImg,scaleFactor=1.32,minNeighbors=5)#detectMultiScale returns rectangles

    return detected_faces, grayImg

#Given a directory below function returns part of gray_img which is face alongwith its label/ID
def laber_training_data(directory):
    faces=[]
    faceID=[]

    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")#Skipping files that startwith .
                continue

            id=os.path.basename(path)#fetching subdirectory names
            img_path=os.path.join(path,filename)#fetching image path
            print("img_path:",img_path)
            print("id:",id)
            test_img=cv2.imread(img_path)#loading each image one by one
            if test_img is None:
                print("Not loaded ")
                continue
            faces_rect,gray_img=detectFace(test_img)#Calling faceDetection function to return faces detected in particular image
            if len(faces_rect)!=1:
               continue #Since we are assuming only single person images are being fed to classifier
            (x,y,w,h)=faces_rect[0]
            roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from grayscale image
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces,faceID


#Below function trains haar classifier and takes faces,faceID returned by previous function as its arguments
def train_classifier(faces,faceID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceID))
    return face_recognizer




#test_img=cv2.imread('C:\\Users\\pc\\Desktop\\deneme2\\trainingImages\\0\\95ee9d16-5ea7-49de-a18b-5a7a5d3737c9.jpg')#test_img path
#faces_detected,gray_img=detectFace(test_img)
#print("faces_detected:",faces_detected)


#Use just 1 times fo training
#faces,faceID=laber_training_data('C:\\Users\\pc\\Desktop\\deneme2\\trainingImages')
#face_recognizer=train_classifier(faces,faceID)
#face_recognizer.write('trainingData.yml')



face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:\\Users\\Gokturk\\Documents\\visionProje\\trainingData.yml')# we are using trainingData.yml file for recognition

name={0:"Gokturk", 1: "Yusuf"}#creating dictionary containing names for each label



cap=cv2.VideoCapture(0)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    faces_detected, gray_img = detectFace(test_img)



    for (x,y,w,h) in faces_detected:
      cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('face detection ',resized_img)
    cv2.waitKey(10)


    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+w, x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
        print("confidence:",confidence)
        print("label:",label)
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=5)
        predicted_name=name[label]

        if confidence > 37:
           cv2.putText(test_img, predicted_name, (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 4)
           # ser = serial.Serial('COM4', '9600')  ## These are IoT part codes, Implementing the arduino.
           # ser.write('gok'.encode())


    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('face recognition ',resized_img)
    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break


cap.release()
cv2.destroyAllWindows