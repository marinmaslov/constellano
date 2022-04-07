from audioop import reverse
import cv2
import numpy as np
from matplotlib import pyplot as plt


img_path = 'img/ursa_major_3.jpg'

# reading the image in grayscale mode
img_rgb = cv2.imread(img_path)
img = cv2.imread(img_path, 0)

# threshold
# ovako cemo ogranicit koje boje piksela ce nam upas u tockice tj. zvijezde koje trazimo:
"""
    The function cv2.threshold works as, if pixel value is greater than a threshold value, it is assigned one value (may be white), else it is assigned another value (may be black)
    First argument is the source image, which should be a grayscale image(done previously).
    Second argument is the threshold value which is used to classify the pixel values. For threshold value, simply pass zero. Then the algorithm finds the optimal threshold value and returns you as the second output, th.
    If Otsu thresholding is not used, th is same as the threshold value you used.
"""
th, threshed = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

plt.imshow(threshed, 'gray')
plt.title('stars')
plt.show()

# findcontours
cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

"""
    Contours can be explained simply as a curve joining all the continuous points (along the boundary), having same color or intensity.
    The contours are a useful tool for shape analysis and object detection and recognition. Contours give better accuracy for using
    binary images. There are three arguments in cv2.findContours() function, first one is source image, second is contour retrieval
    mode, third is contour approximation method. It outputs the contours and hierarchy. Contours is a Python list of all the contours
    in the image. Each individual contour is a Numpy array of (x, y) coordinates of boundary points of the object.
"""

# filter by area
s1 = 3
s2 = 400
xcnts = []
xcnt_areas = []

for cnt in cnts:
    if s1 < cv2.contourArea(cnt) <s2:
        #print("CNTR AREA: " + str(cv2.contourArea(cnt)))
        xcnt_areas.append(cv2.contourArea(cnt))
        xcnts.append(cnt)

#new_img = cv2.drawContours(img_rgb, xcnts, -1, (0, 0, 255), 2)

# printing output
print("\nDots number: {}".format(len(xcnts)))


#for i in range(0, len(xcnts)):
#    print("\n Dot [" + str(i) + "]: " + str(xcnts[i]))

#cv2.imwrite('output/img/dots_img.jpg', new_img)
#cv2.waitKey(0)






# ODREDI NAJCEVU TOCKICU i onda skupi sve tockice koje su te velicine i do 20% manje od nje
xcnt_areas.sort(reverse = True)
biggest_dot = xcnt_areas[0]
print("BIGGEST DOT: " + str(biggest_dot))

smallest_biggest_dot_limit = biggest_dot - biggest_dot*0.92
print("SMALLEST DOT LIMIT: " + str(smallest_biggest_dot_limit))

# filter by area AGAIN
s1 = 3
s2 = biggest_dot
xcnts_new = []

for cnt in cnts:
    if smallest_biggest_dot_limit <= cv2.contourArea(cnt) <= biggest_dot:
        xcnts_new.append(cnt)

new_img = cv2.drawContours(img_rgb, xcnts_new, -1, (0, 255, 0), 2)

# printing output
print("\nDots number: {}".format(len(xcnts)))


#for i in range(0, len(xcnts)):
#    print("\n Dot [" + str(i) + "]: " + str(xcnts[i]))

cv2.imwrite('output/img/dots_img.jpg', new_img)
cv2.waitKey(0)




contour_img = np.zeros_like(img_rgb)
cv2.fillPoly(contour_img, xcnts_new, (0, 255, 0))
cv2.imwrite('output/img/dots_img_contour.jpg', contour_img)
cv2.waitKey(0)







