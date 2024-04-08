import cv2
import numpy as np

# Leitura da Imagem em RGB
image = cv2.imread(r'C:\Users\usuario\Desktop\Test\image.png')

# Transformando para escala de cinza
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Mostrando imagem em escala de cinza
cv2.imshow('Input grayscale image', grayscale_image)

# Obtendo linhas e colunas da imagem
rows, cols = grayscale_image.shape[:2]

# Criando uma matriz com as mesmas dimensões (colunas x linhas) da imagem em escala cinza
output_image = np.zeros((rows, cols), np.uint8)

# Utilizando filtro passa-alta de Sobel
for row in range(1, rows-1):
    for col in range(1, cols-1):
        gx = grayscale_image[row - 1, col - 1] * (-1) + grayscale_image[row, col - 1] * (-2) + \
             grayscale_image[row + 1, col - 1] * (-1) + grayscale_image[row - 1, col + 1] + \
             grayscale_image[row, col + 1] * 2 + grayscale_image[row + 1, col + 1]

        gy = grayscale_image[row - 1, col - 1] * (-1) + grayscale_image[row - 1, col] * (-2) + \
             grayscale_image[row - 1, col + 1] * (-1) + grayscale_image[row + 1, col - 1] + \
             grayscale_image[row + 1, col] * 2 + grayscale_image[row - 1, col + 1]

        output_image[row, col] = (gx**2 + gy**2)**(1/2)

# Mostrando os resultados
cv2.imshow('Sobel image', output_image)
cv2.waitKey(0)