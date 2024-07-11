import cv2
import face_recognition
import pickle
import os
import numpy


# Importing the Employee images.
folderPath = "Images"
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
employeeIds = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    employeeIds.append(os.path.splitext(path)[0])
    # print(path)
    # print(os.path.splitext(path)[0])
print(employeeIds)

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


print("Encoding started.....")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, employeeIds]
# print(encodeListKnown)
print("Encoding Complete")

file = open("EncodeFile.p", "wb")
pickle.dump(encodeListKnownWithIds,file)
file.close()
print("file saved")