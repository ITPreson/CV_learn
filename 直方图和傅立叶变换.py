import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
cv2.calcHist()参数:
    images:输入的图像
    channels:选择图像的通道  如果是BGR格式，就用[0],[1],[2]表示
    mask:掩膜，是一个大小和image一样的np数组，其中把需要处理的部分指定为1，不需要处理的部分指定为0，一般设置为None，表示处理整幅图像
    histSize:使用多少个bin(柱子)，一般为256
    ranges:像素值的范围，一般为[0,255]表示0~255
    后面两个参数基本不用管。注意，除了mask，其他四个参数都要带[]号。
    
'''

'''用plt直接画直方图'''


def read_img():
    img = cv2.imread("img/cat.jpg", 0)
    # 这个函数不需要你找到histogram，直接用
    # img.ravel()将多维数组降至一维
    plt.hist(img.ravel(), 256)  # 参数：数据，条形数
    plt.show()


'''先统计，再用折线图画，适合统计BGR分别的数值分布'''


def read_img1():
    img = cv2.imread("img/cat.jpg")
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        # 画折线图
        plt.plot(hist, color=col)
    plt.show()


# 带mask的histogram
def read_img2():
    img = cv2.imread("img/cat.jpg", 0)
    # 创建mask
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[100:300, 100:400] = 255
    '''显示带mask的图像'''
    # 原图像和mask做一个与操作
    img_mask = cv2.bitwise_and(img, mask)  # 按位与
    # show("wang", img_mask)
    '''直方图统计带mask的图像'''
    plt.hist(img_mask.ravel(), 255, [1, 256])
    plt.show()
    '''折线图图统计带mask的图像'''
    histogram = cv2.calcHist([img], [0], mask, [255], [0, 256])
    plt.plot(histogram, color='b')
    # plt.show()


# 直方图均衡化
#         它可以增强对比度
#         它会丢失一些图像细节
def Equalize():
    cat = cv2.imread('img/cat.jpg', 0)
    equ = cv2.equalizeHist(cat)
    img = np.hstack((cat, equ))
    show('equ_img', img)


# 自适应均衡化
#       它就是分模块进行均衡化，整体均衡化有时候效果会不好
#       局部直方图均衡化
def clahe_equ():
    cat = cv2.imread('img/cat.jpg', 0)
    # clipLimit颜色对比度的阈值， titleGridSize进行像素均衡化的网格大小，即在多少网格下进行直方图的均衡化操作
    clahe = cv2.createCLAHE(2, (8, 8))
    img = clahe.apply(cat)
    show("clahe", img)


'''
    傅立叶变化  cv2.dft()  cv2.idft()
        时域变换到频域
    低通滤波器：只保留低频，会使图像变得模糊（因为边界没了）
    高通滤波器：只保留高频，会使图像细节增强
    高频：变化剧烈的灰度分量，例如边界
    低频：变化缓慢的灰度分量，例如一片大海
    
    做完傅立叶变化后，频率为0的部分会在左上角，通常要把它放到中心的位置，用numpy的shift来做
    并且做完傅立叶变化后，它返回的结果是双通道的（实部和虚部），要转化成图像格式才能显示（0，255）
    
    第一步：载入图片
    第二步：使用np.float32进行格式转换
    第三步：使用cv2.dft进行傅里叶变化
    第四步：使用np.fft.shiftfft将低频转移到中间位置
    第五步：使用cv2.magnitude将实部和虚部投影到空间域
    第六步：进行作图操作
    
    对图像处理时，对他的频谱图进行操作运算要快的多
'''


def fliye():
    lina = cv2.imread('img/lena.jpg',0)
    # 做傅立叶时必须先转化为np.float32格式的，官网强制要求
    lina_float32 = np.float32(lina)
    # 做傅立叶变换
    dft = cv2.dft(lina_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    # 使用np.fft.shiftfft()将变化后的图像的低频转移到中心位置
    fft_shift = np.fft.fftshift(dft)
    # 把它的双通道返回结果转化为图像格式的频谱图
    print(fft_shift.shape)
    print(fft_shift[:,:,1])
    # 使用np.fft.shiftfft()将变化后的图像的低频转移到中心位置;   cv2.magnitude计算矩阵的加和平方根
    magnitude_spectrum = 20 * np.log(cv2.magnitude(fft_shift[:,:,0], fft_shift[:,:,1]))
    plt.subplot(121)
    plt.title("input_img")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(lina,cmap='gray')

    plt.subplot(122)
    plt.title("magnitude_spectrum")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(magnitude_spectrum,cmap='gray')
    # 最后这步才是显示图的！！
    plt.show()

# 利用傅立叶变换对图片进行操作
def idft():
    lena = cv2.imread("img/lena.jpg", 0)
    img = np.float32(lena)
    # 傅立叶变化生成频谱图，并把频率低的放到中间   !!必须加flags =
    dft = cv2.dft(img, flags = cv2.DFT_COMPLEX_OUTPUT)
    fftshift = np.fft.fftshift(dft)
    # 计算中心坐标
    rows,col = img.shape
    crow,ccol = int(rows/2),int(col/2)
    '''
    # 构造低通滤波器，频率低的已经在频谱图的中间
    mask = np.zeros((rows,col,2),np.uint8)
    mask[crow-30:crow+30,ccol-30:ccol+30] = 1
    '''
    # 构造高通滤波器
    mask = np.ones((rows,col,2),np.uint8)
    mask[crow-30:crow+30,ccol-30:ccol+30] = 0

    # IDFT逆傅立叶变换
    fshift = fftshift*mask
    ifftshift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(ifftshift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

    plt.plot(122)
    plt.title("magnitude_spectrum")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img_back, cmap='gray')
    # 最后这步才是显示图的！！
    plt.show()



def show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # read_img()
    # read_img1()
    # read_img2()
    # Equalize()
    # clahe_equ()
    # fliye()
    idft()