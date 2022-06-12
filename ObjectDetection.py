import numpy as np
import cv2


img = cv2.imread('assests/volleyball_Practice.jpg', 0) #Load as grayscale for detection algorithm
template = cv2.imread('assests/pink_Shoe.png', 0)

height, width = template.shape

#These are all of the different methods of doing template matching...
#It is reccomended to paste all of them in and see which one works best
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

#Loop through the methods to see which one works the best
for method in methods:
    #Copies the image so we have a "new" starting image to draw our detections(rectangles) on
    img2 = img.copy() 

    #This line of code essentially slides our template around our base image and see where it matches at
    result = cv2.matchTemplate(img2, template, method) 

    #See explanation behind match template to understand why we are getting these values
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    #If we use these methods, we want to take the min location, otherwise we will take the max
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc

    #Since location is the top left of the rectangle we want to draw, now we need to find the bottom right corner
    #Basically just calculating the size of our template image to match with the base image
    bottom_right = (location[0] + width, location[1] + height)

    cv2.rectangle(img2, location, bottom_right, 255, 2)
    cv2.imshow('Match', img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



