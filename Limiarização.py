import cv2

# Leitura da Imagem RGB
image = cv2.imread(r'C:\Users\usuario\Desktop\Test\image.png')

# Transformando para escala de cinza
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicando Limiarização (Threshold)
ret, threshold_image = cv2.threshold(grayscale_image, 70, 255, cv2.THRESH_BINARY)

# Mostrando imagem em escala de cinza
cv2.imshow('Input grayscale image', grayscale_image)

# Mostrando resultado da limiarização 
cv2.imshow('Threshold result', threshold_image)

cv2.waitKey(0)

# Salvando os resultados
cv2.imwrite('threshold_result.jpg', threshold_image)