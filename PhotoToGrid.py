# Author: Danny Chung
# Date: 4/23/2022
# Description: Takes input of a drawn maze image, recognizing square, triangle, and asterisk shapes.
#              Outputs a computer readable 2D grid in .txt format.
#              Primary usage for generating a grid dungeon game.


import cv2 as cv
import os
import numpy as np
import csv
import time
import sys

class PhotoToGrid:
    def __init__(self, file_path, grid_dimension_x, grid_dimension_y):
        # Input image file.
        self.system_path = os.getcwd()
        self.file = str(self.system_path) + "\\" + str(file_path)
        self.input_not_gray = cv.imread(self.file)
        self.input = cv.cvtColor(self.input_not_gray, cv.COLOR_BGR2GRAY)

        # Dimensions of input image in pixels.
        self.input_width = int(self.input.shape[1])
        self.input_height = int(self.input.shape[0])

        # Dimensions of input box measurement in pixels.
        self.one_box_width = grid_dimension_x
        self.one_box_height = grid_dimension_y

        # Calculate output grid width and height dimensions.
        self.grid_width = (round(self.input_width / self.one_box_width) * 2) + 1
        self.grid_height = (round(self.input_height / self.one_box_height) * 2) + 1

        # Grid as array of arrays.
        self.grid = self.build_grid()

        # Width and Height traversal arrays for usage in image analysis.
        self.width_traversal = self.create_traversal_array(self.input_width, self.one_box_width)
        self.height_traversal = self.create_traversal_array(self.input_height, self.one_box_height)

        # Dictionary of images to reference shape comparisons.
        self.wall_images_dict = self.build_wall_dictionary()

        # Final grid.
        self.final_grid = self.build_final_grid()

    def build_grid(self):
        """
        Builds initial grid based on self.grid_width and self.grid_height.
        Fills perimeter with '#' as borders.
        Returns the initial grid.
        """
        grid = [[" " for x in range(self.grid_width)] for x in range(self.grid_height)]
        for i in range(0, self.grid_height):
            grid[i][0] = '#'
            grid[i][self.grid_width - 1] = '#'
        for i in range(0, self.grid_width):
            grid[0][i] = '#'
            grid[self.grid_height - 1][i] = '#'

        return grid

    def create_traversal_array(self, input_dimension, box_dimension):
        """
        Creates an array to be used in image analysis.
        The array contains pixel increments to help mask the image into analyzable sections
        for each iteration of self.build_final_grid(),
        """
        traversal = []
        section = round(input_dimension / box_dimension) * 2
        for number in range(0, section - 1):
            traversal.append(int((number * (1/section)) * input_dimension))
        return traversal

    def build_final_grid(self):
        """
        Fills in grid with inner walls.
        Places an "S" on the grid where a square is located.
        Places an "E" on the grid where a triangle is located.
        Places an "*" on the grid where an asterisk is located.
        Returns the final grid.
        """
        for y in range(0, len(self.height_traversal)):
            for x in range(0, len(self.width_traversal)):
                # wall_check_toggle flag, is set to 1 if detecting a wall, 0 if detecting a grid square.
                wall_check_toggle = 0
                # Default trim, for grid square analysis, process inner 60% of square to get rid of edge noise.
                trim_start_x, trim_start_y = .20, .20
                trim_end_x, trim_end_y = .80, .80

                # Narrowed trims for wall detection, only process inner 10% of square to get rid of edge noise.
                # Even x and y values mean that a wall is analyzed. Set wall check flag.
                if ((x % 2) != 0) and ((y % 2) == 0):
                    trim_start_x, trim_end_x = .45, .55
                    wall_check_toggle = 1
                # Odd y values also mean that a wall is being analyzed. Set wall check flag.
                if (y % 2) != 0:
                    trim_start_y, trim_end_y = .45, .55
                    wall_check_toggle = 1

                # Otherwise, it is a grid square analysis. Check for shape.
                # Dimensions for analysis of each box.
                start_rectangle = (int((self.one_box_width * trim_start_x)) + (self.width_traversal[x]),
                                   int(self.one_box_height * trim_start_y) + (self.height_traversal[y]))
                end_rectangle = (int((self.one_box_width * trim_end_x)) + (self.width_traversal[x]),
                                 int(self.one_box_height * trim_end_y) + (self.height_traversal[y]))

                input_image = cv.imread(self.file)

                # Transform the image for processing.
                processed_image_portion = self.transform_image(input_image, start_rectangle, end_rectangle)

                # Uncomment the below line to view the image.
                # cv.imshow('Reversed', processed_image_portion)
                # cv.waitKey(0)

                if wall_check_toggle == 1:
                    grid_value = self.check_wall(processed_image_portion)

                elif wall_check_toggle == 0:
                    grid_value = self.check_square(processed_image_portion)

                # Set grid value at x and y in grid.
                self.grid[y+1][x+1] = grid_value

                # Uncomment the below three lines to print the grid in console.
                # for line in self.grid:
                #     print(line)
                # print("")
            # Print loading progress
            sys.stdout.write('\r')
            sys.stdout.write("Loading... %d%%" % int(round((y / len(self.height_traversal)), 2) * 100) )
            #print("Loading... ", int(round((y / len(self.height_traversal)), 2) * 100), "%")
            sys.stdout.flush()

        sys.stdout.write('\r')
        sys.stdout.write("Loading... 100%")
        sys.stdout.flush()
        sys.stdout.write('\r')
        time.sleep(0.5)
        # Return the final grid.
        return self.grid

    def transform_image(self, input_image, start_rectangle, end_rectangle):
        """
        Transform the original image into one that can be analyzed.
        Applies Gaussian Blur, then thresholding, then masking.
        Masking isolates the initial image into analyzable sections, getting rid of everything else in the image.
        """
        # Reduce unmarked grid edges by blurring.
        blur = cv.GaussianBlur(input_image, (11, 11), cv.BORDER_DEFAULT)

        # Thresholding to remove unmarked grid walls.
        threshold, thresh = cv.threshold(blur, 100, 255, cv.THRESH_BINARY)

        # Mask for image processing. Must start at same size of the input image.
        black = np.zeros(input_image.shape[:2], dtype='uint8')
        mask = cv.rectangle(black, start_rectangle, end_rectangle, 255, -1)

        # Bitwise operations on mask and input image.
        masked = cv.bitwise_not(thresh, thresh, mask=mask)
        masked2 = cv.bitwise_and(masked, masked, mask=mask)

        # Final masked image.
        masked3 = cv.bitwise_not(masked2)

        # Uncomment the following 4 lines to view image.
        # temp_mask = cv.bitwise_not(masked)
        # cv.imshow('ISOLATED OBJECT', masked3)
        # cv.imshow('MASK INCREMENTS', temp_mask)
        # cv.waitKey(0)

        return masked3

    def check_wall(self, processed_image_portion):
        """
        Checks the image portion for a wall.
        Returns "#" to be placed in the grid in case a wall is detected, returns " " otherwise.
        """
        wall_confidence = 0

        # Iterate through the two possible walls in the dictionary, and compare the processed image.
        for item in self.wall_images_dict:
            reference_image = self.wall_images_dict[item]
            comparison = cv.matchTemplate(processed_image_portion, reference_image, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(comparison)
            if wall_confidence < max_val:
                wall_confidence = max_val

        if wall_confidence > .2:
            return "#"

        else:
            return " "

    def check_square(self, processed_image_portion):
        """
        Checks the image portion for a symbol in the grid square.
        Returns an "S" if a square is detected.
        Returns an "E" if a triangle is detected.
        Returns an "*" if an asterisk is detected.
        """
        # Convert processed image portion to greyscale.
        img_grey = cv.cvtColor(processed_image_portion, cv.COLOR_BGR2GRAY)

        # Apply Canny edge detection.
        img_canny = cv.Canny(img_grey, 50, 100)

        # Apply dilation to the detected edges.
        kernel = np.ones((2, 2))
        img_dil = cv.dilate(img_canny, kernel, iterations=1)

        # Find contours, if any.
        contours, hierarchy = cv.findContours(img_dil, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # If no contours are found, return " ".
        if len(contours) < 1:
            return ' '

        # Make sure that the area of the contour is significant enough to be detected.
        # Returns " " if the area of the contour is insignificant.
        area = int(cv.contourArea(contours[0]))
        box = int((self.one_box_width * self.one_box_height * .10))
        if area < box:
            return ' '

        # Uncomment the next 2 lines to view the image.
        # cv.imshow("POINTS ANALYSIS", img_dil)
        # cv.waitKey(0)

        # Check how many edges are detected.
        perimeter = cv.arcLength(contours[0], True)
        approx = cv.approxPolyDP(contours[0], 0.02 * perimeter, True)

        # If 1 to 4 edges are detected, return S. (square)
        # If 5 to 8 edges are detected, return E. (triangle)
        # If more than 8 edges are detected, return *. (asterisk)
        # This allows for a margin of error in detecting less-than-ideally drawn shapes.
        if len(approx) > 1 and len(approx) < 5:
            return 'S'
        elif len(approx) > 4 and len(approx) < 9:
            return 'E'
        else:
            return '*'

    def build_wall_dictionary(self):
        """
        Create dictionary of the reference images for walls.
        Returns the dictionary.
        """
        horizontal = str(self.system_path) + "\\" + "horiz_ref.jpg"
        horizontal_ref = cv.imread(horizontal)
        vertical = str(self.system_path) + "\\" + "vert_ref.jpg"
        vertical_ref = cv.imread(vertical)

        dictionary = {'H': horizontal_ref, 'V': vertical_ref}

        return dictionary

    def write_txt(self):
        """
        Creates a .txt file based off of self.final_grid, named "dungeon.txt".
        """
        with open("dungeon.txt", "w", newline=None) as new_file:
            csv_writer = csv.writer(new_file, delimiter=",")
            for row in self.final_grid:
                print(row)
                csv_writer.writerow(row)

        # Get rid of last new line carriage return.
        with open("dungeon.txt", "rb+") as new_file:
            new_file.seek(-3, os.SEEK_END)
            new_file.truncate()


# Testing
p2g = PhotoToGrid('input_image.jpg', 90, 90)
p2g.write_txt()
