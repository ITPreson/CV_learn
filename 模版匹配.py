import cv2
import numpy as np

'''
    0、平方差匹配法TM_SQDIFF  值越小，越相关
    1、归一化平方差匹配法TM_SQDIFF_NORMED  越接近0，越相关
    2、相关匹配法TM_CCORR  值越大，越相关
    3、归一化相关匹配法TM_CCORR_NORMED    越接近1，越相关
    4、相关系数匹配法TM_CCOEFF   值越大，越相关
    5、归一化相关系数匹配法TM_CCOEFF_NORMED 越接近1，越相关
    
'''


def read_img():
    # 这里的0，就代表cv2.IMREAD_GRAYSCALE
    img = cv2.imread("img/lena.jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    template = cv2.imread("img/face.png", 0)
    h, w = template.shape[:2]
    # 这里相当于窗口滑动，每个窗口的结果，一共用（A-a+1）*(B-b+1)
    # 拿模版去和原图匹配，一步一步滑动，给出了这么多结果
    # 这里注意要灰度图
    # 最好用归一化有的算法
    result = cv2.matchTemplate(gray, template, 1)
    # 这里你要看你选的那个算法，可能要大值，也可能要小值
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    img_c = img.copy()
    left_top = min_loc
    right_bottom = (left_top[0] + w, left_top[1] + h)
    rectangle = cv2.rectangle(img_c, left_top, right_bottom, (0, 255, 0), 2)
    show("wang", rectangle)


def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    read_img()
