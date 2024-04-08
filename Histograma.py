import cv2
import matplotlib.pyplot as plt

# Leitura da Imagem em RGB
image = cv2.imread(r'C:\Users\usuario\Desktop\Test\image.png')

# Transformando imagem para escala de cinza
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Equalização da Imagem
equalized_image = cv2.equalizeHist(grayscale_image)

# Calculando Histograma da imagem original e da imagem equalizada, utilizando OpenCV
original_hist = cv2.calcHist(grayscale_image, channels=[0], mask=None, histSize=[256], ranges=[0, 256])
equalized_hist = cv2.calcHist(equalized_image, channels=[0], mask=None, histSize=[256], ranges=[0, 256])

# Mostrando a imagem original, imagem equalizada e seus respectivos histogramas
# Podemos (Em Python) calcular e mostrar o histograma utilizando Matplotlib 
plt.figure(1)
plt.subplot(221)
plt.imshow(grayscale_image, cmap='gray')
plt.subplot(222)
plt.hist(grayscale_image.ravel(), 256, [0, 256])
plt.subplot(223)
plt.imshow(equalized_image, cmap='gray')
plt.subplot(224)
plt.hist(equalized_image.ravel(), 256, [0, 256])
plt.show()