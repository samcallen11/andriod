'''
import face_recognition as fc
from cv2 import imshow as i
image = fc.load_image_file("D:\\PROGRAMMING\\images\\pic4.jpg")
file_locations = fc.face_locations(image)
'''

import face_recognition
known_image = face_recognition.load_image_file("D:\\PROGRAMMING\\images\\pic4.jpg")
unknown_image = face_recognition.load_image_file("D:\\PROGRAMMING\\images\\pic7.jpg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding],
 unknown_encoding)

print(results)