import cv2
import face_recognition
import numpy as np
imgTom = face_recognition.load_image_file('imagesbasics/Tomcruise1.jpg')
imgTom = cv2.cvtColor(imgTom, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('imagesbasics/Tomcruisetest.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgTom)[0]
encodeTom = face_recognition.face_encodings(imgTom)[0]
cv2.rectangle(imgTom,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

facelocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)

result = face_recognition.compare_faces([encodeTom],encodeTest)
faceDis = face_recognition.face_distance([encodeTom],encodeTest)
print(result)
cv2.putText(imgTest,f'{result} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Tomcruise1', imgTom)
cv2.imshow('Tomcruisetest', imgTest)
cv2.waitKey(0)
