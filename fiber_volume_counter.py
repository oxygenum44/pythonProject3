# importing libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("19/19_1_m09.tif")

grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
crop_img = grayImage[:-64, :]
median = cv2.medianBlur(crop_img, 5)
'''
#crop_img = grayImage[:, :]
dst = cv2.blur(crop_img,(3,3))

# Morphological closing
dst = cv2.erode(dst,None,iterations = 3)
dst = cv2.dilate(dst,None,iterations = 3)
a = 1 / 9
kernel = np.array([[a, a, a],
                   [a, a, a],
                   [a, a, a]])
a = -1
kernel2 = np.array([[0.5*a, 1.5*a, 0.5*a],
                    [1.5*a, 9, 1.5*a],
                    [0.5*a, 1.5*a, 0.5*a]])
dst = cv2.filter2D(crop_img, -1, kernel2)
#kernel = np.ones((5, 5), np.float32) / 25
#dst = cv2.filter2D(dst, -1, kernel)

number_of_pix = np.sum(median)

histr = cv2.calcHist([dst], [0], None, [256], [0, 256])
'''
histr2 = cv2.calcHist([crop_img], [0], None, [256], [0, 256])
# print(histr)
# plt.plot(histr / histr.max(), label="median")
plt.plot(histr2/histr2.max(), label="gray")
plt.title("Histograms")
plt.xlim([0, 256])
plt.legend()
plt.show()

THRESHOLD_HUMAN = int(input())

(thresh, blackAndWhiteImage) = cv2.threshold(crop_img, THRESHOLD_HUMAN, 255, cv2.THRESH_BINARY)

# find all your connected components (white blobs in your image)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(blackAndWhiteImage, connectivity=4)
# connectedComponentswithStats yields every seperated component with information on each of them, such as size
# the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
sizes = stats[1:, -1];
nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)
# here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
# Wlokna
min_size = 100

# your answer image
img2 = np.zeros(output.shape, np.uint8)
# for every component in the image, you keep it only if it's above min_size
for i in range(0, nb_components):
    if sizes[i] >= min_size:
        img2[output == i + 1] = 255

img3 = cv2.bitwise_not(img2)

# img3 = np.uint8(img3)
# find all your connected components (white blobs in your image)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img3, connectivity=4)
# connectedComponentswithStats yields every seperated component with information on each of them, such as size
# the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
sizes = stats[1:, -1];
nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)
# here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
min_size = 100

# your answer image
img4 = np.zeros(output.shape, np.uint8)
# for every component in the image, you keep it only if it's above min_size
for i in range(0, nb_components):
    if sizes[i] >= min_size:
        img4[output == i + 1] = 255

img4 = cv2.bitwise_not(img4)

cv2.imshow('img4', img4)
# plt.title('org')
# plt.imshow(img)
# plt.show()
cv2.imshow('BW', blackAndWhiteImage)
cv2.imshow('median', median)
cv2.waitKey()
# plt.title('res')
# plt.imshow(img4)
# plt.show()
'''
r = 60
# remove noise
dst = cv2.blur(img4,(5,5))

# Morphological closing
dst = cv2.erode(dst,None,iterations = 3)
dst = cv2.dilate(dst,None,iterations = 3)
img5 = np.uint8(img4)
circles = cv2.HoughCircles(img5
                           ,cv2.HOUGH_GRADIENT
                           ,2
                           ,param1=30
                           ,param2=60
                           ,minDist=10
                           ,minRadius=20
                           ,maxRadius=80
                           )
print(circles)
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(img5, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(img5, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('detected circles', img5)
#dst2 = cv2.filter2D(img4, -1, kernel2)
#cv2.imshow('result2', dst2)

cv2.waitKey(0)
'''


# counting the number of pixels
number_of_white_pix = np.sum(img4 == 255)
number_of_black_pix = np.sum(img4 == 0)

print('Number of white pixels:', number_of_white_pix)
print('Number of black pixels:', number_of_black_pix)

ratio = 100 * (number_of_white_pix / (number_of_black_pix + number_of_white_pix))
print('Ratio of the white to whole image:', ratio)
