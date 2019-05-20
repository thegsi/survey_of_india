# Thanks to Marguerite Le Riche

import sys
sys.path.append('/usr/local/opt/opencv/lib/python3.7/site-packages')
import cv2
import numpy as np


source_image = './39O16_bands_exp.tif'


# open source image as grayscale with opencv
grayscale = cv2.imread(source_image, 0)

removenoise = cv2.GaussianBlur(grayscale,(5,5),0)
#cv2.imwrite('removenoise.tif', removenoise)
del grayscale

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharp = cv2.filter2D(removenoise, -1, kernel)
sharp2 = cv2.filter2D(sharp, -1, kernel)
cv2.imwrite('sharp.tif', sharp2)
del removenoise

sharp3 = cv2.filter2D(sharp2, -1, kernel)
cv2.imwrite('sharp3.tif', sharp3)

#a global absolute threshold
# Manually setting the value may be an issue if there is variation between your map sheets
ret, dots = cv2.threshold(sharp2, 50, 255, cv2.THRESH_BINARY_INV)
#cv2.imwrite('dots.tif', dots)


#get the contours
_, contours, hierarchy = cv2.findContours(dots, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
        #take out some of the unwanted stuff
        area = cv2.contourArea(contour)

        if area <= 20:
            cv2.drawContours(dots, [contour], -1, 0, -1)
            cv2.drawContours(dots, [contour], -1, 0, 2)


cv2.imwrite('nodots.tif', dots)
del dots

# open source image as colour with opencv
colour = cv2.imread(source_image, 1)
#cv2.imwrite('original.tif', colour)

#white out the definite background
dots = cv2.imread('nodots.tif', 1)
dots2 = cv2.bitwise_not(dots)
neatened = cv2.add(colour, dots2)
cv2.imwrite('neat1.tif', neatened)
del dots
del dots2

#get the darker colour values
#a global absolute threshold
# Manually setting the value may be an issue if there is variation between your map sheets
ret, darkcolours = cv2.threshold(neatened, 80, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite('darkcolours.tif', darkcolours)
del darkcolours

#turn them to grayscale
darkgrays= cv2.imread('darkcolours.tif', 0)
#darkgrays = cv2.cvtColor(cv2.COLOR_BGR2GRAY, darkcolours)
ret, darkthresh = cv2.threshold(darkgrays, 1, 255, cv2.THRESH_BINARY)
cv2.imwrite('darkthresh.tif', darkthresh)
del darkgrays

#and take them out
darkthresh = cv2.imread('darkthresh.tif', 1)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
bigthresh = cv2.dilate(darkthresh, kernel, iterations = 1)
neatened2 = cv2.add(neatened, bigthresh)
cv2.imwrite('neat2.tif', neatened2)
del darkthresh
del bigthresh
del neatened
del neatened2

#get the neatened image as grayscale
grayneat = cv2.imread('neat2.tif', 1)


hsv = cv2.cvtColor(grayneat, cv2.COLOR_BGR2HSV)

# isolate the muddy and reddish colours
boundaries = [([10, 30, 50], [25, 100, 180])]
# loop over the boundaries
for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower)
    upper = np.array(upper)
    # find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(hsv, hsv, mask=mask)

    rgb = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)

    cv2.imwrite('neat3.tif', rgb)



# boundaries2 = [([0, 0, 0], [10, 10, 10])]
# # loop over the boundaries
# for (lower, upper) in boundaries:
#     # create NumPy arrays from the boundaries
#     lower = np.array(lower)
#     upper = np.array(upper)
#     # find the colors within the specified boundaries and apply the mask
#     mask = cv2.inRange(hsv, lower, upper)
#     output = cv2.bitwise_and(hsv, hsv, mask=mask)



finalgray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

ret, neatthresh = cv2.threshold(finalgray, 1, 255, cv2.THRESH_BINARY)
cv2.imwrite('neattresh.tif', neatthresh)

#get the contours
_, contours, hierarchy = cv2.findContours(neatthresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
        #obliterate as much of the the black in the image as possible
        area = cv2.contourArea(contour)
        if area < 20:
            cv2.drawContours(neatthresh, [contour], -1, 0, -1)
            cv2.drawContours(neatthresh, [contour], -1, 0, 2)
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(colour, (x - 100, y - 100), (x + w + 100, y + h + 100), (0, 255, 2255), 10)

cv2.imwrite('spotremoved.tif', neatthresh)
cv2.imwrite('result.png', colour)
