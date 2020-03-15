import cv2
import numpy as np


def read_img():
    img = cv2.imread("img/dige.png", cv2.IMREAD_GRAYSCALE)

    # 腐蚀操作  腐蚀是对白色进行腐蚀
    kernel = np.ones((3, 3), np.uint8)
    erode_img = cv2.erode(img, kernel, iterations=1)

    # 膨胀操作  相对于黑色
    kernel1 = np.ones((3, 3), np.uint8)
    dilate_img = cv2.dilate(erode_img, kernel1, iterations=3)

    # 开运算：先腐蚀，再膨胀。 它只不过是把腐蚀膨胀结合起来了
    kernel2 = np.ones((3, 3), np.uint8)
    open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2)

    # 闭运算：先膨胀，再腐蚀
    kernel3 = np.ones((3, 3), np.uint8)
    close_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel3)

    # 梯度运算：图像当中的减操作，梯度=膨胀-腐蚀
    img1 = cv2.imread("img/pie.png")
    kernel4 = np.ones((3, 3), np.uint8)
    cv2.erode(img1, kernel4, iterations=1)
    cv2.dilate(img1, kernel4, iterations=1)
    gradient_img = cv2.morphologyEx(img1, cv2.MORPH_GRADIENT, kernel4)

    # 礼帽和黑帽
    # 礼帽：原始输入-开运算
    # 黑帽：闭运算-原始输入


    cv2.imshow("open", open_img)
    cv2.imshow("gradient", gradient_img)
    cv2.imshow("close", close_img)
    cv2.imshow("erode", erode_img)
    cv2.imshow("dilate", dilate_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    read_img()
