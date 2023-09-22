import numpy as np
import cv2 as cv
import glob

# Critério de terminação
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Preparação dos pontos do objeto, como (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
objp = np.zeros((6*7, 3), np.float32)
square_size_in_cm = 2.5
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2) * square_size_in_cm

# Arrays para armazenar os pontos do objeto e pontos da imagem de todas as imagens
objpoints = []  # Pontos 3D no espaço do mundo real
imgpoints = []  # Pontos 2D no plano da imagem

images = glob.glob('images/*.jpg')
valid_images = 0
invalid_images = 0
images_amount = len(images)

for fname in images:
    img = cv.imread(fname)
    img_resized = cv.resize(img, (1920, 1080)) #1920x1080/1280x720/960x540
    gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)

    gray = cv.GaussianBlur(gray, (5, 5), 0)  # filtragem de ruído

    #edges = cv.Canny(gray, 100, 200)  # Detecção de bordas com Canny
    #gray = cv.convertScaleAbs(gray, alpha=1.5, beta=0) # equalização do histograma
    #gray = cv.equalizeHist(gray) # ajuste de contraste

    # Encontrar os cantos do tabuleiro de xadrez
    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

    # Se encontrado, adicionar os pontos do objeto e pontos da imagem (após refinamento)
    if ret == True:
        valid_images += 1
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Desenhar e exibir os cantos
        cv.drawChessboardCorners(img_resized, (7, 6), corners2, ret)
        cv.imshow('img', img_resized)
        cv.waitKey(50)

    else:
        print("Imagem inválida.")
        invalid_images += 1

print(f"Images válidas: {valid_images}")
print(f"Imagens inválidas: {invalid_images}")

# Dimensões da imagem
img_height = img_resized.shape[0]
img_width = img_resized.shape[1]

# Dimensões da imagem em pixels
x_pixels = corners2[-1, 0, 0] - corners2[0, 0, 0]
y_pixels = corners2[-1, 0, 1] - corners2[0, 0, 1]

# Conversão para metros
x_meters = x_pixels * (square_size_in_cm / img_width)
y_meters = y_pixels * (square_size_in_cm / img_height)

# Exibir as dimensões em pixels e metros
print("Dimensões da imagem (pixels):")
print("Largura (x):", x_pixels)
print("Altura (y):", y_pixels)

print("Dimensões da imagem (metros):")
print("Largura (x):", x_meters)
print("Altura (y):", y_meters)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# Exibição dos parâmetros obtidos
print("Matriz de Câmera (Intrínsecos):\n", mtx)
distancia = np.linalg.norm(tvecs[0]) / 100
print("Distância da câmera ao objeto:", distancia, "metros")

# Fator de escala (assumindo que o vetor de translação está em relação à escala do objeto)
scale_factor = square_size_in_cm / img_width

# Vetor de translação em metros
tvecs_meters = tvecs[0] * scale_factor

# Distância da câmera à imagem em metros
distance_meters = np.linalg.norm(tvecs_meters)

# Exibir a distância em metros
print("Distância da câmera à imagem:", distance_meters, "metros")

# "Undistortion" de uma imagem (remoção de distorções)
img = cv.imread('images/WIN_20230704_20_44_37_Pro.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)

cv.destroyAllWindows()
