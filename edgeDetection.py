import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("./images/lena.tif",0)

#Roberts
kernelRx = np.matrix([[1,0],[0,-1]])
kernelRy = np.matrix([[0,1],[-1,0]])
robertsx = cv2.filter2D(img,-1,kernelRx)
robertsy = cv2.filter2D(img,-1,kernelRy)
robertsxAbs = np.absolute(robertsx)
robertsyAbs = np.absolute(robertsy)
robertsx8u = np.uint8(robertsxAbs)
robertsy8u = np.uint8(robertsyAbs)
roberts = cv2.addWeighted(robertsx8u,.5,robertsy8u,.5,0)
plt.subplot(141),plt.imshow(img,cmap = 'gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(robertsx8u,cmap = 'gray'),plt.title('Roberts 1')
plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(robertsy8u,cmap = 'gray'),plt.title('Roberts 2')
plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(roberts,cmap = 'gray'),plt.title('Roberts')
plt.xticks([]), plt.yticks([])
plt.show()

#Prewitt
kernelPx = np.matrix([[1,0,-1],[1,0,-1],[1,0,-1]])
kernelPy = np.matrix([[1,1,1],[0,0,0],[-1,-1,-1]])
prewittx = cv2.filter2D(img,-1,kernelPx)
prewitty = cv2.filter2D(img,-1,kernelPy)
prewittxAbs = np.absolute(prewittx)
prewittyAbs = np.absolute(prewitty)
prewittx8u = np.uint8(prewittxAbs)
prewitty8u = np.uint8(prewittyAbs)
prewitt = cv2.addWeighted(prewittx8u,.5,prewitty8u,.5,0)
plt.subplot(141),plt.imshow(img,cmap = 'gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(prewittx8u,cmap = 'gray'),plt.title('Prewitt X')
plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(prewitty8u,cmap = 'gray'),plt.title('Prewitt Y')
plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(prewitt,cmap = 'gray'),plt.title('Prewitt')
plt.xticks([]), plt.yticks([])
plt.show()

#Sobel
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
sobelxAbs = np.absolute(sobelx)
sobelyAbs = np.absolute(sobely)
sobelx8u = np.uint8(sobelxAbs)
sobely8u = np.uint8(sobelyAbs)
sobel = cv2.addWeighted(sobelx8u,.5,sobely8u,.5,0)
plt.subplot(141),plt.imshow(img,cmap = 'gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(sobelx8u,cmap = 'gray'),plt.title('Sobel X')
plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(sobely8u,cmap = 'gray'),plt.title('Sobel Y')
plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(sobel,cmap = 'gray'),plt.title('Sobel')
plt.xticks([]), plt.yticks([])
plt.show()


#Laplacian
laplacian = cv2.Laplacian(img,cv2.CV_64F)
laplacianAbs = np.absolute(laplacian)
laplacian8u = np.uint8(laplacianAbs)
plt.subplot(121),plt.imshow(img,cmap = 'gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(laplacian8u,cmap = 'gray'),plt.title('Laplacian')
plt.xticks([]), plt.yticks([])
plt.show()

np.set_printoptions(threshold=np.inf)

#Canny
canny = cv2.Canny(img, 100, 200)
plt.subplot(121),plt.imshow(img,cmap = 'gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(canny,cmap = 'gray'),plt.title('Canny')
plt.xticks([]), plt.yticks([])
plt.show()
