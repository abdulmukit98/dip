import numpy as np
import cv2 as cv
import scipy.ndimage as ndi

cap = cv.VideoCapture('cat.mp4')
angle = 0

cap.set(cv.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 144)

if not cap.isOpened():
    print("Can't open video file.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame... Exiting...")
        break

    cv.imshow('Original Video', frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('Grayscale Video', gray)

    angle = (angle+1)%360
    output = ndi.rotate(frame,angle,reshape=False)
    cv.imshow('Rotated Frame', output)

    if cv.waitKey(1) == 27:  
        break

cap.release()
cv.destroyAllWindows()
