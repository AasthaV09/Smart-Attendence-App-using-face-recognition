from typing import List

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

from numpy import ndarray

path = 'imageattendence'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cls in myList:
    currImg = cv2.imread(f'{path}/{cls}')
    images.append(currImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendence(name):
    with open('Attendence.csv','r+') as f:
        myDataList = f.readlines()
        namelist = []
        for line in myDataList:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrFrame = face_recognition.face_locations(imgS)
    encodesCurrFrame = face_recognition.face_encodings(imgS,facesCurrFrame)

    for encodeface,faceloc in zip(encodesCurrFrame,facesCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeface)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeface)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceloc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendence(name)


    cv2.imshow('Webcam',img)
    cv2.waitKey(1)



#faceloc = face_recognition.face_locations(imgTom)[0]
#encodeTom = face_recognition.face_encodings(imgTom)[0]
#cv2.rectangle(imgTom,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

#facelocTest = face_recognition.face_locations(imgTest)[0]
#encodeTest = face_recognition.face_encodings(imgTest)[0]
#cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)

#result = face_recognition.compare_faces([encodeTom],encodeTest)
#faceDis = face_recognition.face_distance([encodeTom],encodeTest)