import cv2
import os
import pickle
import numpy as np
import face_recognition
import cvzone

cap = cv2.VideoCapture(0)  # Change to 0 if 1 doesn't work
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread(r"D:\Face_Detection\Resources\background.png")

# Importing the mode images into a list
folderModePath = r"D:\Face_Detection\Resources\Modes"
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))
    
# print(len(imgModeList))

#Load the encoding file
print("Loading Encode File....")
file = open("EncodeFile.p","rb")
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, employeeIds  = encodeListKnownWithIds 
# print(employeeIds)
print("Encode File Loaded")



if imgBackground is None:
    print("Error: Could not read the image file. Check the path and file integrity.")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame from camera.")
        break
     
    imgS = cv2.resize(img,(0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)
      
    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[1]
    
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        # print("matches",matches)
        # print("faceDistance",faceDis)
        
        matchIndex = np.argmin(faceDis)
        # print("Match Index",matchIndex)
        
        if matches[matchIndex]:
            # print("Known Face Detected")
            # print(employeeIds[matchIndex])
            y1, x2,  y2, x1 = faceLoc
            y1, x2,  y2, x1 = y1 * 4, x2 * 4,  y2 * 4, x1 * 4
            bbox = 55+x1, 162+y1, x2-x1, y2-y1
            imgBackground = cvzone.cornerRect(imgBackground,bbox,rt=0)
    
    # cv2.imshow("Webcam", img)
    cv2.imshow("Face Attendance", imgBackground)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



