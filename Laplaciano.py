import cv2

# Leitura da imagem em RGB
image = cv2.imread(r'C:\Users\usuario\Desktop\Test\image.png')

# Transformando a imagem para escala de cinza
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicando filtro passa-alta Laplaciano
laplace = cv2.Laplacian(grayscale_image, ddepth=cv2.CV_64F, ksize=3)

# convertendo para uint8
laplace = cv2.convertScaleAbs(laplace)

# Equalizando a imagem do filtro Laplaciano
equalized_laplacian = cv2.equalizeHist(laplace)

# Mostrando imagem inicial em escala cinza
cv2.imshow('Input grayscale image', grayscale_image)

# Mostrando resultados do filtro Laplaciano
cv2.imshow('Laplacian filter result', laplace)

# Mostrando resultados do filtro Laplaciano após equalização
cv2.imshow('Equalized Laplacian', equalized_laplacian)

cv2.waitKey(0)