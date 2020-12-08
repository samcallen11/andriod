import cv2

def face_detected(detected, img):
    for (x, y, width, height) in detected:
        return img[y:y+height,x:x+width]

path = 'download.jfif'  #img adress
original_image = cv2.imread(path)

if original_image is not None:
    img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
  
    detected_faces = face_cascade.detectMultiScale(image=img, scaleFactor=1.3, minNeighbors=4)
    original_image = face_detected(detected_faces, original_image)
    # as inja bebad berat niaz nist faghat baraye namayeshe(midonam midoni vali hala)
    cv2.imshow('main', original_image)
    cv2.waitKey(1) 
