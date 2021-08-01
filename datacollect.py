
import cv2
import numpy as np


face_classifier = cv2.CascadeClassifier(r'C:\Users\91866\PycharmProjects\Face-Recognition-Using-Transfer-Learning\haarcascade_frontalface_default.xml')


def face_extractor(img):

    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    

    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        cropped_face = img[y:y+h+50, x:x+w+50]

    return cropped_face


cap = cv2.VideoCapture(0)
count = 0


while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (600, 600))

        file_name_path = r'C:\Users\91866\PycharmProjects\Face-Recognition-Using-Transfer-Learning\images\Train\Rahul/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)


        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 50:
        break
count=0
while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (600, 600))

        cv2.putText(face, 'camera images', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)

    if cv2.waitKey(1) == 13 or count == 100:
        break

count=0
while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (600, 600))

        file_name_path = r'C:\Users\91866\PycharmProjects\Face-Recognition-Using-Transfer-Learning\images\Train\RahulCamera/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 50:
        break

cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete")








