

from os import listdir
from os.path import isfile, join,isdir
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import os

class DataPrepare():
    def __init__(self):
        self.basedir=r'./DeepLabv3FineTuning/ADEChallengeData2016/'
        self.basedir_annt=r'..\freespacetrainingdataset\ADEChallengeData2016\annotations\training'
        self.basedir_imgt=r'..\freespacetrainingdataset/ADEChallengeData2016/images/training'
        #'C:\Users\caofamily\temp\freespacetrainingdataset\ADEChallengeData2016

        self.target_imgdir=r'.\FloorDataL\Images'
        self.target_segdir=r'.\FloorDataL\Masks'

        self.names=[]
        for i in range(1, 20000, 1): # Just to generate bigger numbers
            self.names.append([f"{i:05d}.png",f"{i:05d}.jpg"])
    def get_img_seg_files(self):
        segfiles = list(Path(self.basedir_annt).rglob("*.png"))
        imgfiles=list(Path(self.basedir_imgt).rglob("*.jpg"))
        return imgfiles,segfiles
    
    def process(self,segno=4,size=(320,240)):
        imgfs,segfs=self.get_img_seg_files()
        n_files = len(segfs)
        processed=0
        satisfied=0

        for segf in segfs:
            try:
                processed+=1
                seg=np.array(Image.open(segf))
                if segno in np.unique(seg):
                    satisfied+=1
                    seg[seg!=segno]=0
                    seg[seg==segno]=255
                    imgr,segr=self.process_img_seg(segf,seg,size)
                    self.write_to_target(imgr,segr)

                sys.stdout.write('\r>> Converting image %d/%d satisfied %d' % (processed, n_files, satisfied))
                sys.stdout.flush()
            except Exception as e:
                print(segf,e)
            
            #if satisfied>150:
                #break
    
    def process_img_seg(self,segf,seg,size):
        head,tail=os.path.split(segf)
        imgf=tail.replace('.png','.jpg')
        imgf=os.path.join(self.basedir_imgt,imgf)
        img=np.array(Image.open(imgf))
        seg=cv2.resize(seg,size,interpolation=cv2.INTER_NEAREST)
        img=cv2.resize(img,size,interpolation=cv2.INTER_NEAREST)
        return img,seg
    
    def write_to_target(self,img,seg):
        segf,imgf=self.names.pop(0)
        imgf=os.path.join(self.target_imgdir,imgf)
        segf=os.path.join(self.target_segdir,segf)
        cv2.imwrite(segf,seg)
        cv2.imwrite(imgf,img)
    






if __name__ == '__main__':
    dp=DataPrepare()
    dp.process(segno=4,size=(320,240))
       

