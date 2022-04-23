import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

file1 = os.path.join(os.path.dirname(__file__), 'blank.jpg')
blank1 = cv.imread(file1)

file2 = os.path.join(os.path.dirname(__file__), 'square.jpg')
square = cv.imread(file2)

file3 = os.path.join(os.path.dirname(__file__), 'horiz.jpg')
horiz = cv.imread(file3)

file4 = os.path.join(os.path.dirname(__file__), 'vert.jpg')
vert = cv.imread(file4)

file5 = os.path.join(os.path.dirname(__file__), 'horiz_ref.jpg')
horiz_ref = cv.imread(file5)

file6 = os.path.join(os.path.dirname(__file__), 'vert_ref.jpg')
vert_ref = cv.imread(file6)

# canny = cv.Canny(square, 125, 175)
# cv.imshow("edges", canny)
#
# contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
#
# for contour in contours:
#     print("SQUARE", contour)


field1 = square.copy()
field2 = horiz.copy()
field3 = vert.copy()

# #Find horizontal in horizontal field.
# find_horiz = horiz_ref.copy()
# result1 = cv.matchTemplate(field2, find_horiz, cv.TM_CCOEFF_NORMED)
# cv.imshow("Result", result1)
# min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result1)
# print("Best match confidence %s" % (max_val))
#
# #Should not find horizontal in blank field.
# result2 = cv.matchTemplate(field1, find_horiz, cv.TM_CCOEFF_NORMED)
# cv.imshow("Should not find horiz", result2)
# min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result2)
# print("Best match confidence %s" % max_val)
#
# #Should not find horizontal in vertical field.
# result3 = cv.matchTemplate(field3, find_horiz, cv.TM_CCOEFF_NORMED)
# cv.imshow("Should not find horiz in vert", result3)
# min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result3)
# print("Best match confidence %s" % max_val)
#
#
#
# #Should not find vertical in horizontal field.
# find_vert = vert_ref.copy()
# result1 = cv.matchTemplate(field2, find_vert, cv.TM_CCOEFF_NORMED)
# cv.imshow("Result2", result1)
# min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result1)
# print("Best match confidence %s" % (max_val))
#
# #Should not find vertical in blank field.
# result2 = cv.matchTemplate(field1, find_vert, cv.TM_CCOEFF_NORMED)
# cv.imshow("Should not find vert", result2)
# min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result2)
# print("Best match confidence %s" % max_val)
#
# #Should find vertical in vertical field.
# result3 = cv.matchTemplate(field3, find_vert, cv.TM_CCOEFF_NORMED)
# cv.imshow("Should vert in vert", result3)
# min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result3)
# print("Best match confidence %s" % max_val)


test_file = os.path.join(os.path.dirname(__file__), 'test_grid.jpeg')
test_grid = cv.imread(test_file)

#Reduce edges by blurring
blur = cv.GaussianBlur(test_grid, (21, 21), cv.BORDER_DEFAULT)
cv.imshow("Blurred", blur)

#Thresholding
threshold, thresh = cv.threshold(blur, 150, 255, cv.THRESH_BINARY)
cv.imshow('Simple Thresholded', thresh)

field = thresh.copy()
cv.imshow('Field', field)

#MASKING
#Mask must start at same size of the image.
black = np.zeros(field.shape[:2], dtype='uint8')
#Draw a rectangle over the blank and call it mask.
mask = cv.rectangle(black, (45, 0), (135, 90), 255, -1)
cv.imshow("MASK", mask)
masked = cv.bitwise_not(field, field, mask=mask)
cv.imshow('Masked Image', masked)
masked2 = cv.bitwise_and(masked, masked, mask=mask)
cv.imshow('Not Masked Image', masked2)
masked3 = cv.bitwise_not(masked2)
cv.imshow('Reversed', masked3)

#IMAGE CHECKS
# #Should find vertical.
find_vert = vert_ref.copy()
result1 = cv.matchTemplate(masked3, find_vert, cv.TM_CCOEFF_NORMED)
cv.imshow("Found", result1)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result1)
print("Best match confidence %s" % (max_val))

# #Should not find horizontal.
find_horiz = horiz_ref.copy()
result2 = cv.matchTemplate(masked3, find_horiz, cv.TM_CCOEFF_NORMED)
cv.imshow("Should not find horiz", result2)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result2)
print("Best match confidence %s" % max_val)

# #Should not find blank.
find_blank = blank1.copy()
result3 = cv.matchTemplate(masked3, find_blank, cv.TM_CCOEFF_NORMED)
cv.imshow("Should not find blank", result3)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result3)
print("Best match confidence %s" % max_val)


cv.waitKey(0)