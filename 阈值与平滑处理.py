import cv2
import numpy as np


def read_img():
    # 阈值操作
    img = cv2.imread("img/cat.jpg", cv2.IMREAD_GRAYSCALE)
    ret, img1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # 大于127，=255，小于127，=0
    ret, img2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)  # 和上面相反
    ret, img3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)  # truncate截断，大于127，就等于127，小于，就不变
    ret, img4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)  # 和上面相反
    cv2.imshow("wang", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_img1():
    # 平滑处理（滤波器）
    img = cv2.imread("img/lenaNoise.png")

    # 均值滤波 （默认进行归一化操作）
    blur_img = cv2.blur(img, (3, 3))  # 滤波器的核大小一般都是奇数

    # 方框滤波，它和均值滤波一样，就是多了一个可选的是否要进行归一化操作   -1这个位置表示通道数，这个值是你不知道，电脑自动计算的最后填充的，
    box_filter = cv2.boxFilter(img, -1, (3, 3), normalize=False)

    # 高斯滤波  离锚点越近的，权重应当越高.  它的卷积核里面的权重数值符合高斯分布，更重视中间的
    gaussian_img = cv2.GaussianBlur(img, (5, 5), 1)

    # 中值滤波  中值模板的卷积对去除椒盐噪声有比较好的作用
    median_img = cv2.medianBlur(img, 5)

    # 把所有图像拼接在一起展示
    # np.vstack():在竖直方向上堆叠
    # np.hstack():在水平方向上平铺
    ret = np.hstack((blur_img, gaussian_img, median_img))

    cv2.imshow("wei", ret)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    read_img()
