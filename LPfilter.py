import cv2
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

img = cv2.imread("E:\\imagecorrect\\practice1_crop\\23.jpg", 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
rows , cols = img.shape
crow , ccol = int(rows/2) , int(cols/2)
#
mask = np.zeros((rows , cols ) , np.uint8)
mask[crow-30:crow+30 , ccol-30:ccol+30] = 1
l_shift = fshift.copy()
h_shift = fshift.copy()
thredshold = fshift.copy()
l_shift[crow-30:crow+30 , ccol-30:ccol+30] = 0
h_shift = fshift * mask

#
l_ishift = np.fft.ifftshift(l_shift)
h_ishift = np.fft.ifftshift(h_shift)
ishift = np.fft.ifftshift(fshift)

#
iimg = np.fft.ifft2(ishift)
l_img = np.fft.ifft2(l_shift)
h_img = np.fft.ifft2(h_shift)
#
iimg = np.abs(iimg)
l_img = np.abs(l_img)
h_img = np.abs(h_img)

plt.imsave("E:\\imagecorrect\\practice1_crop_FFT\\23.jpg",l_img,cmap='gray')

plt.subplot(221) , plt.imshow(img , cmap = 'gray')
plt.title('original') , plt.axis('off')
#
plt.subplot(222) , plt.imshow(l_img , cmap = 'gray')
plt.title('l_img') , plt.axis('off')
#
plt.subplot(223) , plt.imshow(h_img , cmap = 'gray')
plt.title('h_img') , plt.axis('off')
#
plt.subplot(224) , plt.imshow(iimg , cmap = 'gray')
plt.title('iimg') , plt.axis('off')
plt.show()

