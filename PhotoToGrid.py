import cv2
import cv2 as cv
import os
import numpy as np

class PhotoToGrid:
    def __init__(self, file_path, grid_dimension_x, grid_dimension_y):
        # Input image file.
        self.file = file_path
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
        self.reference_images_dict = self.build_square_dictionary()
        self.wall_images_dict = self.build_wall_dictionary()

        # Final grid.
        self.final_grid = self.build_final_grid()

    def printPath(self):
        pass

    def build_grid(self):
        # Build border of grid as walls '#'
        grid = [[" " for x in range(self.grid_width)] for x in range(self.grid_height)]
        for i in range(0, self.grid_height):
            grid[i][0] = '#'
            grid[i][self.grid_width - 1] = '#'
        for i in range(0, self.grid_width):
            grid[0][i] = '#'
            grid[self.grid_height - 1][i] = '#'
        return grid

    def create_traversal_array(self, input_dimension, box_dimension):
        traversal = []
        section = round(input_dimension / box_dimension) * 2
        for number in range(0, section - 1):
            traversal.append(int((number * (1/section)) * input_dimension))
        return traversal

    def build_final_grid(self):
        for y in range(0, len(self.height_traversal)):
            for x in range(0, len(self.width_traversal)):
                wall_check_toggle = 0
                # Default trims.
                trim_start_x, trim_start_y = .20, .20
                trim_end_x, trim_end_y = .80, .80
                # Narrowed trims for wall detection.
                if ((x % 2) != 0) and ((y % 2) == 0):
                    trim_start_x, trim_end_x = .45, .55
                    wall_check_toggle = 1
                if ((y % 2) != 0):
                    trim_start_y, trim_end_y = .45, .55
                    wall_check_toggle = 1
                #Otherwise, check for shape.
                start_rectangle = (int((self.one_box_width * trim_start_x)) + (self.width_traversal[x]),
                                   int((self.one_box_height)* trim_start_y) + (self.height_traversal[y]))
                end_rectangle = (int((self.one_box_width * trim_end_x)) + (self.width_traversal[x]),
                                    int((self.one_box_height) * trim_end_y) + (self.height_traversal[y]))

                input = cv.imread(self.file)

                processed_image_portion = self.transform_image(input, start_rectangle, end_rectangle)

                #cv.imshow('Reversed', processed_image_portion)

                if wall_check_toggle == 1:
                    grid_value = self.check_wall(processed_image_portion)

                elif wall_check_toggle == 0:
                    grid_value = self.check_square(processed_image_portion)

                self.grid[y+1][x+1] = grid_value

                for line in self.grid:
                    print(line)

                print("")

                #cv.waitKey(0)

    def transform_image(self, input, start_rectangle, end_rectangle):
        # Reduce unmarked grid edges by blurring.
        blur = cv.GaussianBlur(input, (11,11), cv.BORDER_DEFAULT)
        # Thresholding to remove unmarked grid.
        threshold, thresh = cv.threshold(blur, 170, 255, cv.THRESH_BINARY)
        # Mask for image processing. Must start at same size of the input image.
        black = np.zeros(input.shape[:2], dtype='uint8')
        mask = cv.rectangle(black, start_rectangle, end_rectangle, 255, -1)
        # Bitwise operations on mask and input image.
        masked = cv.bitwise_not(thresh, thresh, mask=mask)
        masked2 = cv.bitwise_and(masked, masked, mask=mask)
        masked3 = cv.bitwise_not(masked2)

        temp_mask = cv.bitwise_not(masked)

        cv.imshow('ISOLATED OBJECT', masked3)
        cv.imshow('MASK INCREMENTS', temp_mask)
        cv.waitKey(0)

        return masked3

    def check_wall(self, processed_image_portion):
        wall_confidence = 0
        for item in self.wall_images_dict:
            reference_image = self.wall_images_dict[item]
            comparison = cv.matchTemplate(processed_image_portion, reference_image, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(comparison)
            #print(item, "Confidence match", item, ": ", max_val)
            if wall_confidence < max_val:
                wall_confidence = max_val

        if wall_confidence > .2:
            return "#"
        else:
            return " "

    def check_square(self, processed_image_portion):
            imgGrey = cv.cvtColor(processed_image_portion, cv2.COLOR_BGR2GRAY)

            imgCanny = cv.Canny(imgGrey, 50, 100)

            kernel = np.ones((2, 2))
            imgDil = cv.dilate(imgCanny, kernel, iterations=1)

            contours, hierarchy = cv.findContours(imgDil, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            if len(contours) < 1:
                return ' '

            area = int(cv.contourArea(contours[0]))
            box = int((self.one_box_width * self.one_box_height * .10))

            if area < box:
                print(cv.contourArea(contours[0]), "AREA!")
                print(self.one_box_width * self.one_box_height * .05, "BOX!")
                print (area, "<", box)
                return ' '

            cv.imshow("POINTS ANALYSIS", imgDil)
            cv.waitKey(0)

            perimeter = cv.arcLength(contours[0], True)
            approx = cv.approxPolyDP(contours[0], 0.02 * perimeter, True)
            print(len(approx))

            if len(approx) > 1 and len(approx) < 5:
                return 'S'

            elif len(approx) > 4 and len(approx) < 9:
                return 'E'

            else:
                return '*'

    def build_square_dictionary(self):
        square = os.path.join(os.path.dirname(__file__), 'sq_ref.jpg')
        sq_ref = cv.imread(square)
        circle = os.path.join(os.path.dirname(__file__), 'dot_ref.jpg')
        dot_ref = cv.imread(circle)
        triangle = os.path.join(os.path.dirname(__file__), 'tri_ref.jpg')
        tri_ref = cv.imread(triangle)

        dictionary = {}
        dictionary['S'] = sq_ref
        dictionary['E'] = tri_ref
        dictionary['.'] = dot_ref

        return dictionary

    def build_wall_dictionary(self):
        horizontal = os.path.join(os.path.dirname(__file__), 'horiz_ref.jpg')
        horizontal_ref = cv.imread(horizontal)
        vertical = os.path.join(os.path.dirname(__file__), 'vert_ref.jpg')
        vertical_ref = cv.imread(vertical)

        dictionary = {}
        dictionary['H'] = horizontal_ref
        dictionary['V'] = vertical_ref

        return dictionary

p2g = PhotoToGrid('C:/Users/ffoxxttrott/PycharmProjects/OpenCV/test_draw.jpg', 120, 90)
p2g.printPath()
