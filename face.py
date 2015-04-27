#coding=utf-8
import cv2
import cv2.cv as cv
import numpy as np
import math
 
def detect(img, cascade):
    '''detectMultiScale函数中smallImg表示的是要检测的输入图像为smallImg，
faces表示检测到的人脸目标序列，1.3表示每次图像尺寸减小的比例为1.3，
 4表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸),
 CV_HAAR_SCALE_IMAGE表示不是缩放分类器来检测，而是缩放图像，Size(20, 20)为目标的最小最大尺寸'''
    rects = cascade.detectMultiScale(img, scaleFactor=1.3,
                                    minNeighbors=5, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    print rects
    return rects
 
#在img上绘制矩形
def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
      # now calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
      # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
      # calculate the move from the old center to the new center combined
      # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
      # the move only affects the translation, so update the translation
      # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

class FaceDetect(object):
    def __init__(self):
        #正脸
        self.front_fn = 'haarcascades/haarcascade_frontalface_alt.xml'
        # cascade_fn = 'lbpcascades/lbpcascade_frontalface.xml'
        #侧脸
        self.profile_fn = 'haarcascades/haarcascade_profileface.xml'
        # cascade_fn = 'lbpcascades/lbpcascade_profileface.xml'

        #读取分类器,CascadeClassifier下面有一个detectMultiScale方法来得到矩形
        self.frontCascade = cv2.CascadeClassifier(self.front_fn)
        self.profileCascade = cv2.CascadeClassifier(self.profile_fn)
        return

    def detect(self, img):
        """
        :param img:{numpy}
        :return:
        """
        #vis为img副本
        vis = img.copy()

        #转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #直方图均衡处理
        gray = cv2.equalizeHist(gray)

        #通过分类器得到rects
        rects = detect(gray, self.frontCascade)
        if len(rects) == 0:
            #侧脸检测
            rects = detect(gray, self.profileCascade)
            if len(rects) == 0:
                #镜像 在侧脸检测
                gray = cv2.flip(gray, 1)
                rects = detect(gray, self.profileCascade)
                vis = cv2.flip(vis, 1)

        #画矩形
        draw_rects(vis, rects, (0, 255, 0))
        return vis

if __name__ == '__main__':
    img = cv2.imread("face/1.jpg")
    model = FaceDetect()
    vis = model.detect(img)

    cv2.imshow('facedetect', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()