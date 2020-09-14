import cv2
import os
import numpy as np


def detectFace(test_img):
    grayImg = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  # convert color image to grayscale
    haar_cascade = cv2.CascadeClassifier(
        'C:\\Users\\Gokturk\\Downloads\\haarcascade_frontalface_default.xml')  # Load haar classifier
    detected_faces = haar_cascade.detectMultiScale(grayImg, scaleFactor=1.32,
                                                   minNeighbors=5)  # detectMultiScale returns rectangles

    return detected_faces, grayImg



def laber_training_data(directory):
    faces = []
    faceID = []

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")  # Skipping files that startwith .
                continue

            id = os.path.basename(path)  # fetching subdirectory names
            img_path = os.path.join(path, filename)  # fetching image path
            print("img_path:", img_path)
            print("id:", id)
            test_img = cv2.imread(img_path)  # loading each image one by one for training
            if test_img is None:
                print("Not loaded ")  ## if the image cant load properly
                continue
            faces_rect, gray_img = detectFace(
                test_img)  # Calling faceDetection function to return faces detected in particular image
            if len(faces_rect) != 1:
                continue  # Asuming just 1 person face each picture
            (x, y, w, h) = faces_rect[0]
            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from grayscale image
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces, faceID


# classifier our images for recognition
def train_classifier(faces, faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer


# ######################/**/-# ##########################################################################

test_img = cv2.imread('C:\\Users\\Gokturk\\Downloads\\WhatsApp Image 2020-05-21 at 14.06.57.jpeg')  # MG
# test_img = cv2.imread('C:\\Users\\Gokturk\\Pictures\\5fc9655c-5b62-4d0e-b12e-5f58c361c39d.jpg')  # Gokturk
# test_img = cv2.imread('C:\\Users\\Gokturk\\Downloads\\WhatsApp Image 2020-05-21 at 02.12.04.jpeg')  # Yusuf
faces_detected, gray_img = detectFace(test_img)
print("faces_detected:", faces_detected)
# Use 1 times for train images and create yml file. After the training dont use
# faces, faceID = laber_training_data('C:\\Users\\pc\\Desktop\\deneme2\\trainingImages')
# face_recognizer = train_classifier(faces, faceID)
# face_recognizer.write('trainingData.yml')

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:\\Users\\Gokturk\\Documents\\visionProje\\trainingData.yml')

name = {0: "Gokturk", 1: "Yusuf"}

for face in faces_detected:
    (x, y, w, h) = face
    roi_gray = gray_img[y:y + h, x:x + h]
    label, confidence = face_recognizer.predict(roi_gray)  # predict the person
    print("confidence:", confidence)  # How confidence the images and faces
    print("label:", label)
    (x, y, w, h) = face
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=5)
    predicted_name = name[label]
    if(confidence > 37):  # If confidence more than 37 then don't print predicted face text on screen
        continue
    cv2.putText(test_img, predicted_name, (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 4)

    resized_img = cv2.resize(test_img, (500, 500))
    cv2.imshow("Recognition", resized_img)
    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

resized_img = cv2.resize(test_img, (700, 700))
cv2.imshow("Regular detection", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows
