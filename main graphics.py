# importing cv2 and numpy
import cv2
import numpy as np

# Reading the image
image = cv2.imread('2dgraphics_convolution_ex1.png')

# Creating the kernel - edge detection
kernel2 = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])
kernel3 = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])

# Applying convolution with depth -1 and defined kernel
img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)
img2 = cv2.filter2D(src=image, ddepth=-1, kernel=kernel3)

# Shoeing the original and output image
cv2.imshow('Original', image)
cv2.imshow('Kernel Blur', img)
cv2.imshow('Sharpen', img2)

cv2.waitKey()
cv2.destroyAllWindows()

kernel_erode = np.ones((5,5), np.uint8)
eroded = cv2.erode(image, kernel_erode, iterations=5)
cv2.imshow('Original', image)
cv2.imshow("Eroded", eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel_dilate = np.ones((5,5), np.uint8)
dilated = cv2.dilate(image, kernel_dilate, iterations=5)
cv2.imshow('Original', image)
cv2.imshow("Dilated", dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
# Define lower and uppper limits of what we call "brown"
# lower mask (0-10)
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)
mask = mask0+mask1


# Change image to red where we found brown
image[mask>0]=(0,0,0)
"""
image = cv2.imread('sky.jpg')
kernel_open = np.ones((5,5), np.uint8)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_open)
cv2.imshow('Original', image)
cv2.imshow("opening", opening)
cv2.waitKey(0)
cv2.destroyAllWindows()