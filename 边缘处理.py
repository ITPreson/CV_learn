import cv2

# 梯度的方向跟边缘的方向是垂直的
# 非极大值抑制：比较锚点在梯度方向上与紧挨着的两个点的梯度幅值大小，比他俩都大，留下
'''
    Canny算法：一共分四步，平滑处理，梯度和方向，非极大值抑制，双阈值检测
    
    平滑处理这里用的高斯滤波器，
    梯度这里用的Sobel算子、方向用arctan(y/x),
    双阈值检测：设置一个最大阈值、一个最小阈值。梯度值大于最大阈值，留下；小于最小阈值，丢弃；在中间的看有没有和已留下的相连线
              可以看出，最小阈值设定的越小，要求就越低，边缘线越多,就更细致
    
'''


def read_img():
    img = cv2.imread("img/lena.jpg", cv2.IMREAD_GRAYSCALE)

    canny = cv2.Canny(img, 50, 100)

    cv2.imshow("wang", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    read_img()
