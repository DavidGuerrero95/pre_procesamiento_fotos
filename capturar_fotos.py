from pdiFun import *

# Lectura de la cámara:
cam = cv2.VideoCapture(1)


# Eigir lectura a resolusión de 720 HD
def make_720p():
    cam.set(3, 1290)  # Ancho en pixeles
    cam.set(4, 720)  # Alto  en pixeles


make_720p()
title = 'foto'
name = 0
while True:
    ret, img = cam.read()
    img = cv2.GaussianBlur(img, (7, 7), 0)

    if cv2.waitKey(0) == 99:
        cv2.imwrite(title + str(name) + '.PNG', img)
        print('tecla', name)
        name = name + 1
        print('save')

    if cv2.waitKey(1) == 27:  # Salida con Esc
        break
