import cv2 
import numpy as np
import os 

path = "zelena glava"
orb = cv2.ORB_create(nfeatures=1000)
images = []
classNames = []
mylist = os.listdir(path)
print = ("Total Classes Detected: ",len(mylist))
for cl in mylist:
    ImgCurrent = cv2.imread(f"{path}/{cl}",0)
    classNames.append(os.path.splitext(cl)[0])
def findDes(images):
    desList = []
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        desList.append(des)
    return desList


def findID(img,desList):
    kp2,des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1

    for des in desList:
        matches  = bf.knnMatch(des,des2,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        matchList.append(len(good))
     

desList = findDes(images)

cap = cv2.VideoCapture(0)

while True:
    success,img2 = cap.read()
    imgOrginal  = img2.copy()
    img2 = cv2.cvtColor(img2.cv2.COLOR_BGR2GRAY)


    findID(img2,desList)

    cv2.imshow("img2",imgOrginal)
    cv2.waitKey(1)