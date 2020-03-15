import cv2
import numpy as np

'''

    图像金字塔： 高斯金字塔、拉普拉斯金字塔
    
    高斯金字塔：
        向下采样方法（缩小）
                       将图像与一个高斯核进行卷积，重要的是步长为2（把偶数行、列去掉）
                  向上采样方法（扩大）
        上采样、下采样都会损失一些信息，所以先上采样，再下采样之后的图像和原始图像是不一样的，相比较下是会比较模糊 
    
    拉普拉斯金字塔：
        用原始图像减去一个（原始图像先向下，在向上采样的图像），一直循环这个过程下去。             

'''


def read_img():
    img = cv2.imread("img/AM.png")

    up = cv2.pyrUp(img)  # 金字塔上下的采样过程都是2倍的形式
    down = cv2.pyrDown(up)
    img1 = np.hstack((img, down))

    # 拉普拉斯金字塔：做一个示例，只做了一层
    img2 = img - cv2.pyrUp(cv2.pyrDown(img))

    cv2.imshow("wang", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


'''
    轮廓检测：轮廓是一个整体，他不是一个个零散的边缘，它是连在一起的
    cv2.findContours(img,mode,method)
    找轮廓的步骤：1、灰度图   2、变二值图像  3、找轮廓   4、画轮廓   5、显示
'''


# findContours()有三个参数：输入图像，层次类型和轮廓逼近方法
# 该函数会修改原图像，建议使用img.copy()作为输入
# 由函数返回的层次树很重要，cv2.RETR_TREE会得到图像中轮廓的整体层次结构，以此来建立轮廓之间的‘关系’。
# 如果只想得到最外面的轮廓，可以使用cv2.RETE_EXTERNAL。这样可以消除轮廓中其他的轮廓，也就是最大的集合
# 该函数有三个返回值：修改后的图像，图像的轮廓，它们的层次
def find_contours():
    # 记得原始图像即便是黑白图，它也大概率是3通道的，要把它变为灰度图
    img = cv2.imread("img/contours.png", cv2.IMREAD_GRAYSCALE)
    # 变二值图像
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # 找轮廓
    # img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # 由于opencv版本问题，现在只返回两个值，第一个值没有了
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # 绘制轮廓
    img_c = img.copy()
    # 参数：在哪个图像上画，找到的轮廓，轮廓id（一般写-1，就全画了），颜色（B,G,R）,线条粗度
    draw_contours = cv2.drawContours(img_c, contours, 2, (0, 255, 0), 3)

    show("wang", draw_contours)

    '''
        轮廓特征：计算面积，周长等
        cv2.findContours返回的contours是一个轮廓列表，我们需要拿出具体的一个进行计算
    '''
    cnt = contours[0]
    area = cv2.contourArea(cnt)  # 轮廓面积
    length = cv2.arcLength(cnt, True)  # 计算轮廓周长，第二个参数是轮廓是否闭合，默认为True
    print(area, length)


# 轮廓近似
def find_contours_2():
    img = cv2.imread("img/contours2.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img_c = img.copy()
    cnt = contours[0]
    # 颜色这里，你在灰度图上画颜色，当然画不出来了。。。
    draw_contours = cv2.drawContours(img_c, [cnt], -1, (0, 0, 255), 2)
    # show("contours", draw_contours)

    # epsilon在计算机中是极小值的意思
    # 这里为近似轮廓找到一个阈值，一般取轮廓周长的0.1倍
    img_c = img.copy()
    # 注意：这里的0.1不要放在前面乘
    epsilon = cv2.arcLength(cnt, True) * 0.1
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    draw_contours1 = cv2.drawContours(img_c, [approx], -1, (0, 0, 255), 2)
    show("contours", draw_contours1)


# 边界矩阵（画框，人脸识别框框可能原理就是这）
def rectangle():
    img = cv2.imread("img/contours.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours[3]

    # 传进来一个轮廓，画一个矩形边框
    # 这个方法是返回一个x、y坐标，还有它的宽、高
    x, y, w, h = cv2.boundingRect(cnt)
    # 这个是画框操作，通过两个对角点来画
    rec = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    show("rectangle", rec)


def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # read_img()
    # find_contours()
    # find_contours_2()
    rectangle()
