import cv2 as cv
import os
import numpy as np

#Get input image file.
test_file = os.path.join(os.path.dirname(__file__), 'test_grid.jpeg')
test_grid = cv.imread(test_file)

#Get dimensions of input grid.
input_width = int(test_grid.shape[1])
input_height = int(test_grid.shape[0])

#Get dimensions of one grid square.
one_grid_width = 90
one_grid_height = 90

#Calculate grid width and height.
grid_width = (round(input_width/one_grid_width) * 2) + 1
grid_height = (round(input_height/one_grid_height) * 2) + 1
#Create grid as array of arrays.
grid = [[" " for x in range(grid_width)] for x in range(grid_height)]
#Build border of grid as walls '#'
for i in range(0,grid_height):
    grid[i][0] = '#'
    grid[i][grid_width-1] = '#'
    grid[0][i] = '#'
    grid[grid_height-1][i] = '#'

#Sections needed in a row
sections_width = round(input_width/one_grid_width) * 2
sections_height = round(input_height/one_grid_height) * 2

#Reduce unmarked grid edges by blurring.
blur = cv.GaussianBlur(test_grid, (21, 21), cv.BORDER_DEFAULT)

#Thresholding to remove unmarked grid.
threshold, thresh = cv.threshold(blur, 150, 255, cv.THRESH_BINARY)
field = thresh.copy()
cv.imshow('Field', field)

#Widths to traverse.
width_traversal = []
for number in range(0, sections_width):
    width_traversal.append((int((number * (1/sections_width)) * input_width)))
#Heights to traverse.
height_traversal = []
for number in range(0, sections_height):
    height_traversal.append((int((number * (1/sections_height)) * input_height)))
print(width_traversal)
print(height_traversal)

#MASKING
#Mask must start at same size of the input image.
black = np.zeros(field.shape[:2], dtype='uint8')
#Mask should be at size of first grid.
start_rectangle = (int((one_grid_width * .25)) + 44, int((one_grid_height * .25)))
end_rectangle = (int((one_grid_width * .75)) + 44, int((one_grid_height * .75)))
#Mask modified middle 50% of first grid.

#Draw a rectangle over the blank and call it mask.
# mask = cv.rectangle(black, start_rectangle, end_rectangle, 255, -1)
mask = cv.rectangle(black, (66, 22), (111, 67), 255, -1)
cv.imshow("MASK", mask)

black2 = np.zeros(field.shape[:2], dtype='uint8')

mask2 = cv.rectangle(black2, (22, 22), (67, 67), 255, -1)
cv.imshow("MASK2", mask2)


file5 = os.path.join(os.path.dirname(__file__), 'horiz_ref.jpg')
horiz_ref = cv.imread(file5)

file6 = os.path.join(os.path.dirname(__file__), 'vert_ref.jpg')
vert_ref = cv.imread(file6)

cv.waitKey(0)