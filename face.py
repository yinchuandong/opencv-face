#coding=utf-8
import cv2
import cv2.cv as cv
 
img = cv2.imread("face/7.jpg")
 
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
        #转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #直方图均衡处理
        gray = cv2.equalizeHist(gray)
        #通过分类器得到rects
        rects = detect(gray, self.frontCascade)
        #vis为img副本
        vis = img.copy()
        #画矩形
        draw_rects(vis, rects, (0, 255, 0))
        return vis

model = FaceDetect()
vis = model.detect(img)

cv2.imshow('facedetect', vis)
cv2.waitKey(0)
cv2.destroyAllWindows()