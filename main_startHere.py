import cv2
import numpy as np

def nothing(x):
    pass

def initializeTrackbars(intialTracbacVal=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars",360, 240)
    cv2.createTrackbar("Threshold1","Trackbars",intialTracbacVal,255,nothing)
    cv2.createTrackbar("Threshold2","Trackbars", intialTracbacVal,255,nothing)

def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1","Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2","Trackbars")
    src= Threshold1,Threshold2
    return src

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    #loop through countours list
    for i in contours:
        area = cv2.contourArea(i)
        #discard areas below treshold
        if area > 2000:
            #calculates a contour perimeter -> float
            peri = cv2.arcLength(i, True)
            #calcualtes a curve or a polygon with another curve/polygon with less vertices
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            # print(f'Area: {area}, Peri: {peri}, Approx: {approx}')
            # is it a rectangle?
            if area > max_area and len(approx) == 4:
                biggest = approx
                #overwrite max_area for regions that are lareger
                max_area = area
    return biggest,max_area

def reorder(myPoints):
    try:
        myPoints = myPoints.reshape((4 ,2))
        myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
        add = myPoints.sum(1)
 
        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] =myPoints[np.argmax(add)]
        diff = np.diff(myPoints, axis=1)
        myPointsNew[1] =myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]
    
        return myPointsNew  
    except ValueError:
        pass
 
def drawRectangle(img,biggest,thickness):
    try:
        cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 155, 0), thickness)
        cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 155, 0), thickness)
        cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 155, 0), thickness)
        cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 155, 0), thickness)
    
        return img
    except TypeError:
        pass


########################################################################
webCamFeed = True  # set to false if no webcam available
pathImage = "source\\Images\\image004.png"
# main webcam -> 0
cap = cv2.VideoCapture(0)
cap.set(10, 160)
heightImg = 480
widthImg = 640

########################################################################

initializeTrackbars(125)
count = 0

while True:
    # input is either webcam or image
    if webCamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)
    # RESIZE IMAGE
    img = cv2.resize(img, (widthImg, heightImg))
    # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    # CONVERT IMAGE TO GRAY SCALE
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # ADD GAUSSIAN BLUR
    thres = valTrackbars()  # GET TRACK BAR VALUES FOR THRESHOLDS
    imgCanny = cv2.Canny(imgBlur, thres[0], thres[1])  # APPLY CANNY BLUR
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)  # APPLY DILATION
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  # APPLY EROSION
    imgContours =img.copy()

    contours,hierarchy =cv2.findContours(imgThreshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours,contours,-125,(0,255,0),10) 

    biggest,area = biggestContour(contours)
    newPoints = reorder(biggest)
    drawRect= drawRectangle(imgContours,newPoints,5)
    
    pts1 =np.float32(newPoints)

    pts2 =np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])

    try:

        matrix = cv2.getPerspectiveTransform(pts1,pts2)
    except cv2.error:
        pass

    try:
        imgWarp =cv2.warpPerspective(imgContours,matrix,(widthImg,heightImg))
    except NameError:
        pass

    #cv2.imshow("1. Original", img)
    #cv2.imshow("2. Grayscale", imgGray)
    #cv2.imshow("3. Blur", imgBlur)
    #cv2.imshow("4. Canny", imgCanny)
    #cv2.imshow("5. Dilate", imgDial)
    #cv2.imshow("6. Treshold", imgThreshold)
    cv2.imshow("7. imgContours", imgContours)
    try:
        cv2.imshow("8. warp", imgWarp)
    except NameError:
        pass

    imgEroded = cv2.erode (imgDial,kernel, iterations=2)



    # Press x  on keyboard to  exit
    # Close and break the loop after pressing "x" key
    if cv2.waitKey(1) & 0XFF == ord('x'):
        break  # exit infinite loop

     # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print("saving")
        # save image to folder using cv2.imwrite()
        cv2.imwrite("source/Scanned/myImage"+str(count)+".jpg", imgWarp)
        cv2.waitKey(300)
        count += 1
# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
