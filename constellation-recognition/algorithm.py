from audioop import reverse
from sqlite3 import Row
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


# ------------------- FUNKCIJE


def gamma_trans(img, gamma):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img,gamma_table)


# -------------------















img_path = 'img/ursa_major/ursa_major_007.jpg'

# reading the image in grayscale mode
img_rgb = cv2.imread(img_path)
img = cv2.imread(img_path, 0)




# BLURAJ SLIKU DA PONIŠTIŠ ŠUM BAR MALO
ksize = (2, 2)
blr_img = cv2.blur(img, ksize) 

# POVEĆAJ KONTRAST I SMANJI EKSPOZICIJU DA UBIJEŠ SVIJETLOSNO ZAGAĐENJE
# KONTRAST


alpha = 2.5 # Contrast control (1.0-3.0)
beta = 0 # Brightness control (0-100)

adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# EKSPOZICIJA
#adjusted=gamma_trans(adjusted, 10)

plt.imshow(adjusted, 'gray')
plt.title('CONTRAST')
plt.show()

# threshold
# ovako cemo ogranicit koje boje piksela ce nam upas u tockice tj. zvijezde koje trazimo:
"""
    The function cv2.threshold works as, if pixel value is greater than a threshold value, it is assigned one value (may be white), else it is assigned another value (may be black)
    First argument is the source image, which should be a grayscale image(done previously).
    Second argument is the threshold value which is used to classify the pixel values. For threshold value, simply pass zero. Then the algorithm finds the optimal threshold value and returns you as the second output, th.
    If Otsu thresholding is not used, th is same as the threshold value you used.
"""
th, threshed = cv2.threshold(adjusted, 220, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

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
r, c = adjusted.shape

# filter by area
s1 = 0
s2 = 0

if r > c:
    s1 = int(0.0001*r)
    s2 = int(0.2*r)
else:
    s1 = int(0.0002*c)
    s2 = int(0.2*c)


xcnts = []
xcnt_areas = []

for cnt in cnts:
    if s1 < cv2.contourArea(cnt) < s2:
        #print("CNTR AREA: " + str(cv2.contourArea(cnt)))
        xcnt_areas.append(cv2.contourArea(cnt))
        xcnts.append(cnt)

while len(xcnts) > 50:
    s1 = s1 + 1
    s2 = s2 - 1
    xcnts = []
    xcnt_areas = []
    for cnt in cnts:
        if s1 < cv2.contourArea(cnt) < s2:
            #print("CNTR AREA: " + str(cv2.contourArea(cnt)))
            xcnt_areas.append(cv2.contourArea(cnt))
            xcnts.append(cnt)


new_img = cv2.drawContours(img_rgb, xcnts, -1, (255, 255, 255), 2)



# printing output
print("\nDots number: {}".format(len(xcnts)))


for i in range(0, len(xcnts)):
    print("\n Dot [" + str(i) + "]: " + str(xcnts[i]))

cv2.imwrite('output/img/dots_img.jpg', new_img)
cv2.waitKey(0)






# ODREDI NAJCEVU TOCKICU i onda skupi sve tockice koje su te velicine i do 20% manje od nje
xcnt_areas.sort(reverse = True)
biggest_dot = xcnt_areas[0]
print("BIGGEST DOT: " + str(biggest_dot))

smallest_biggest_dot_limit = biggest_dot - biggest_dot*0.4
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
contour_img = cv2.bitwise_not(contour_img)
cv2.fillPoly(contour_img, xcnts_new, (0, 0, 0))
cv2.imwrite('output/img/dots_img_contour.jpg', contour_img)
cv2.waitKey(0)





# Apply MARK on stars
final_img_path = 'output/img/dots_img_contour.jpg'
mark_path = 'img/mark.png'

final_img = cv2.imread(final_img_path)
final_img_bw = cv2.imread(final_img_path, 0)
mark = cv2.imread(mark_path)

rows, cols, _ = final_img.shape
new_rows = int((1000 * cols) / rows)
dimensions = (new_rows, 1000)
final_img = cv2.resize(final_img, dimensions)
final_img_bw = cv2.resize(final_img_bw, dimensions)

#mark = cv2.resize(mark, (40, 40))

image2 = final_img.copy()
image3 = final_img.copy()

gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(cv2.bitwise_not(gray), 180, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)





for c in cnts:
    cv2.drawContours(image2, [c], -1, (0, 255, 0), 2)

    #--- finding bounding box dimensions of the contour ---
    x, y, w, h = cv2.boundingRect(c)
    print(x, y, w, h)

    #--- overlaying the monkey in place of pentagons using the bounding box dimensions---
    #image3[y: y+h, x: x+h] = mark
    image3[y: y + w*5, x: x + w*5] = cv2.resize(mark, (np.abs(x - (x + w*5)), np.abs(y - ( y + w*5))))

cv2.imwrite('output/img/dots_img_contour_final1.jpg', image2)
cv2.imwrite('output/img/dots_img_contour_final2.jpg', image3)

cv2.waitKey(0)
cv2.destroyAllWindows()