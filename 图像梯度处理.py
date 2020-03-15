import cv2
import numpy as np


def read_img():
    # img = cv2.imread("img/pie.png")
    img = cv2.imread("img/lena.jpg", cv2.IMREAD_GRAYSCALE)
    # 梯度处理，Sobel算子,有变化的地方就有梯度（数值增加或减少),感觉和边界检测有点像，额，好像就是用来边缘提取的

    '''
        Sobel算子：水平方向：右边-左边
                  竖直方向：下边-上边
                  减的值不会为负数，它会有截断操作（<0,就全设置为0）
    '''
    '''
     ddepth表示图像深度，一般写-1，表示输出和原图像一致。
     dx，dy表示水平、竖直方向
     ksize表示Sobel算子kernel大小
     
    '''

    # CV_64F is the same as CV_64FC1.   CV_64FC1   64F代表每一个像素点元素占64位浮点数，通道数为1
    # CV_64F可以表示负数，写-1直接截断为0，下面绝对值也没用
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # 这步完了只有半边轮廓，右半边被截断操作，设为了0，为黑色
    scale_absx = cv2.convertScaleAbs(sobel_x)  # 负数被绝对值后，为正了，为白色

    '''
        y方向上
    '''
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    scale_absy = cv2.convertScaleAbs(sobel_y)

    # x、y各加权重后相加
    sobel_xy = cv2.addWeighted(scale_absx, 0.5, scale_absy, 0.5, 0)

    xy_img = np.hstack((scale_absx, scale_absy, sobel_xy))

    cv2.imshow("sobel", xy_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_img1():
    img = cv2.imread("img/lena.jpg", cv2.IMREAD_GRAYSCALE)

    # Scharr算子是Sobel算子的强化，边缘特征更明显
    # 核大小默认是3*3
    scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharrx = cv2.convertScaleAbs(scharrx)
    scharry = cv2.convertScaleAbs(scharry)
    scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)

    cv2.imshow("scharr", scharrxy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    read_img()
    # read_img1()
