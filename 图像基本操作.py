import cv2
import numpy
import matplotlib.pyplot as plt


def read_img():
    # 读取图片时变灰度图
    img = cv2.imread("img/cat.jpg", cv2.IMREAD_GRAYSCALE)
    # cat = img[0:200, 0:200]  # 截取部分图像
    # print(img.shape)
    cv2.imshow("wang", img)
    # <=0是等待，按任意键退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_img1():
    cat = cv2.imread("img/cat.jpg")
    dog = cv2.imread("img/dog.jpg")

    # 颜色通道提取,opencv默认格式是BGR，不是RGB
    '''
    b, g, r = cv2.split(img)
    # 通道合并
    img1 = cv2.merge((b, g, r))
    
    '''

    # 举个例子，只保留b通道. bgr,b的索引号是0
    '''
    c_img = img.copy()  # 没有在原图片上改
    c_img[:, :, 1] = 0  # 把所有的g改为0
    c_img[:, :, 2] = 0  # 把所有的r改为0
    
    '''

    # 边界填充 cv2.copyMakeBorder()方法
    '''
    top_size, left_size, right_size, bottom_size = 290, 207, 207, 290
    # cv2.BORDER_REPLICATE边界直接复制图片边缘相应的尺寸
    replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
    reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT101)
    
    '''

    # 数值相加，会把所有像素点的rgb数值全部加一个数。 +号：如果超过255，会和256取余。 cv2.add()超过255，就会写最大值255
    '''
    img1 = img+10
    '''

    # 图像的放大、缩小
    '''
    resize_img = cv2.resize(img, (0, 0), fx=5, fy=5) # (0,0)是你不知道原始图像的大小，用零来填充，fx、fy表示分别在x、y方向上放大的倍数
    '''

    # 图像的融合
    # 例：猫狗融合
    # 本质：图像相加，但相加的要求是像素大小，也就是矩阵大小必须相同，所以要resize
    # print(cat.shape)
    re_dog = cv2.resize(dog, (500, 414))
    # 但是不能就简单的相加，要有权重和偏置  img = aX+bY+c
    img1 = cv2.addWeighted(cat, 0.4, re_dog, 0.6, 0)

    cv2.imshow("wei", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_video():
    vc = cv2.VideoCapture("video/test.mp4")
    if vc.isOpened():
        ret, frame = vc.read()  # 相当于读取视频的第一帧
    while ret:
        ret, frame = vc.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ret:
            cv2.imshow("wang", gray)
            # 27是esc的ASCII码值
            if cv2.waitKey(10) == 27:
                break


if __name__ == '__main__':
    # read_video()
    read_img1()
