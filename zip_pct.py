# coding:utf-8
from PIL import Image
import os

def compressImage(srcPath, dstPath):
    for filename in os.listdir(srcPath):
        if not os.path.exists(dstPath):
            os.makedirs(dstPath)

        srcFile = os.path.join(srcPath, filename)
        dstFile = os.path.join(dstPath, filename)
        print(srcFile)
        print(dstFile)

        if os.path.isfile(srcFile):
            sImage=Image.open(srcFile)
            w = 64
            h = 64
            dImage=sImage.resize((w, h), Image.ANTIALIAS)
            dImage.save(dstFile)
            print(dstFile + "compressed succeeded")

        if os.path.isdir(srcFile):
            compressImage(srcFile, dstFile)


if __name__ == '__main__':
    compressImage('D:/yc_projects/data/images/All_Images', 'D:/yc_projects/data/images/done_dcgan')
