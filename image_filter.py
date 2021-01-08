import numpy as np
import cv2
import matplotlib.pyplot as plt

f=plt.figure(figsize=(80,40))
imagea=np.load(r"./samples/img1.npy")
oimage=cv2.rotate(imagea,cv2.cv2.ROTATE_90_CLOCKWISE)

target_size = (320, 240)
imagea = cv2.resize(oimage, dsize=target_size)

f.add_subplot(231)
plt.imshow(imagea)
 

target_size = (320, 240)
imagea = cv2.resize(oimage, dsize=target_size)
frontfilter=np.full((3,3),0)
#frontfilter[1,0],frontfilter[1,1]=-1,1
frontfilter[1,0],frontfilter[1,2]=-5,5
frontfilter[0,1],frontfilter[2,1]=-5,5
imagea=cv2.filter2D(imagea,-1,frontfilter)

f.add_subplot(232)
plt.imshow(imagea)


target_size = (320, 240)
imagea = cv2.resize(oimage, dsize=target_size)
frontfilter=np.full((3,3),0)
#frontfilter[1,0],frontfilter[1,1]=-1,1
frontfilter[1,0],frontfilter[1,2]=-5,5
frontfilter[0,1],frontfilter[2,1]=5,-5
imagea=cv2.filter2D(imagea,-1,frontfilter)

f.add_subplot(233)
plt.imshow(imagea)



target_size = (320, 240)
imagea = cv2.resize(oimage, dsize=target_size)
frontfilter=np.full((3,3),0)
#frontfilter[1,0],frontfilter[1,1]=-1,1
frontfilter[1,0],frontfilter[1,2]=5,-5
frontfilter[0,1],frontfilter[2,1]=5,-5
imagea=cv2.filter2D(imagea,-1,frontfilter)

f.add_subplot(234)
plt.imshow(imagea)

target_size = (320, 240)
imagea = cv2.resize(oimage, dsize=target_size)

kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])

G_x = cv2.filter2D(imagea,-1, kernel_x ) 

# Plot them!
ax1 = f.add_subplot(235)

# Actually plt.imshow() can handle the value scale well even if I don't do 
# the transformation (G_x + 255) / 2.
ax1.imshow((G_x + 255) / 2, cmap='gray'); ax1.set_xlabel("Gx")

target_size = (320, 240)
imagea = cv2.resize(oimage, dsize=target_size)
kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
G_y = cv2.filter2D(imagea,-1, kernel_y) 
ax2 = f.add_subplot(236)
plt.imshow(G_y)
#ax2.imshow((G_y + 256) / 2, cmap='gray'); ax2.set_xlabel("Gy")

plt.show()
 