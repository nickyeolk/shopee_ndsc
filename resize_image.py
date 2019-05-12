import os
import cv2
from skimage import transform
import sys
import matplotlib

ROOTDIR='./data/data_raw_20190302/'
def converter(OLDDIR, NEWDIR):
    imgnames=os.listdir(ROOTDIR+OLDDIR)
    for ii, filename in enumerate(imgnames):
        img=cv2.imread(ROOTDIR+OLDDIR+filename)[:,:,::-1]
        img=transform.resize(img, (224, 224))
        matplotlib.image.imsave(ROOTDIR+NEWDIR+filename, img[:,:,::-1])
        if (ii%1000==0):
            print(ii)
converter(sys.argv[1], sys.argv[2])


