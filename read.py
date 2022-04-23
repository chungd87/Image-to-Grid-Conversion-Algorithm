import cv2 as cv
import os
import numpy as np

#GENERATE BLANK IMAGE
blank = np.zeros((500,500,3), dtype='uint8') #image datatype | width, height, channels
#Paint the image a color
blank[:] = 0,0,225
#Put text
cv.putText(blank, "Hellow", (0,225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,225,0), 2)
cv.imshow("Color", blank)

#READ IMAGE
file_name = os.path.join(os.path.dirname(__file__), 'draft.png')

img = cv.imread(file_name)

cv.imshow('Draft', img)

#RESCALE
def rescaleFrame(frame, scale = 0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)

resized_image = rescaleFrame(img)

cv.imshow('Draft2', resized_image)

#RESIZE IGNORING ASPECT RATIO
resized = cv.resize(img, (500,500))
cv.imshow("Resized", resized)

#GREYSCALE
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

#FIND EDGES
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny Edges', canny)
#Can reduce edges by blurring
blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
canny2 = cv.Canny(blur, 125, 175)
cv.imshow("Blurred", canny2)
#Thicken edges
dilated = cv.dilate(canny, (7, 7), iterations = 3)
cv.imshow("Dilated", dilated)
#Crop
cropped = img[100:150, 150:500]
cv.imshow("Cropped", cropped)

#FIND CONTOURS - Around boundaries
##First convert to gray.
countours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
#Looks at edges and returns contours list of coordinates, and hierarchies.
# RETR_LIST returns all contours. RETR_EXTERNAL returns only the outside edges.
# RETR_TREE returns all hierarchical contours (like square in a circle)
# CHAIN_APPROX_NONE gets all coordinates. APPROX_SIMPLE returns two points (of say a line)
print(len(countours), "number of contours")
canny = cv.Canny(blur, 125, 175)
countours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(len(countours), "number of contours")

#THRESHOLD converts to binary form, pure black and white only.
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)


#MASKING
#Mask must start at same size of the image.
blank = np.zeros(img.shape[:2], dtype='uint8')
cv.imshow('Blank Image', blank)
#Draw a circule over the blank and call it mask.
mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -2)
cv.imshow("MASK", mask)
masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Masked Image', masked)


#THRESHHOLDING / BINARIZING IMAGES
#Grayscale first
#Simple Thresholding - compares each pixel to 150 treshold, and sets it to
#255, otherwise it sends it to 0.
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
cv.imshow('Simple Thresholded', thresh)
#Could create an inverse threshold image off this.
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow('Inverse Thresholded', thresh)

#Adaptive Thresholding
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
cv.imshow("Adaptive Thresholded", adaptive_thresh)


#EDGE DETECTION
#Start with Gray
#Laplacian
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow("Laplacian", lap)

#Sabel
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
cv.imshow("Sobel X", sobelx)
cv.imshow("Sobel Y", sobely)
combined_sobel = cv.bitwise_or(sobelx, sobely)
cv.imshow("SobelXY", combined_sobel)


cv.waitKey(0)
