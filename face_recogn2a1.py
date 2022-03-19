import face_recognition
import pyttsx3
import cv2
from cv2 import CascadeClassifier
import numpy as np
import speech_recognition as sr
import shutil
from PIL import Image
# import pyautogui
import time
import sys
import os
from threading import Timer

timeout = 10
t = Timer(timeout, os._exit, [1])

nameNew=""
count_of_unknowns=0
flag1=False
# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
faceCascade = CascadeClassifier('haarcascade_frontalface_default.xml')
known_face_names = []
images = []
known_face_encodings = []

f = open('base.txt', 'r')
known_face_names = f.read().splitlines()
f.close()
i = 0
print(known_face_names)
while i < len(known_face_names):
    images.append(face_recognition.load_image_file("BasePhoto/" + known_face_names[i] + ".jpg"))
    known_face_encodings.append(face_recognition.face_encodings(images[i])[0])
    i += 1
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
engine = pyttsx3.init()
mic = sr.Recognizer()
name = "me"
count_of_unknowns = 0
def dialog():
    global nameD, count_of_unknowns
    count_of_frames_this_face = 0
    #while j<len(face_names):
    name_i = face_names[0]
    if name_i != "Unknown":
        count_of_unknowns=0
        nameD=name_i
        if ";" in name_i:
            nameD = name_i.split(';')[0]
        print("Привет, "+ nameD)
        engine.say("Привет, "+ nameD)
        engine.runAndWait()
    if name_i == "Unknown":
        count_of_unknowns+=1
    if name_i == "Unknown" and count_of_unknowns >= 5:
        print("Я тебя не знаю, как тебя зовут?")
        engine.say("Я тебя не знаю, как тебя зовут?")
        engine.runAndWait()
        nameD=name_i
        with sr.Microphone() as source:
            message = mic.listen(source)
            while nameD=="Unknown":
                try:
                    nameD = mic.recognize_google(message, language="ru-RU")
                    print(nameD)
                except sr.UnknownValueError:
                    print("Я не расслышал, повторите")
                    engine.say("Я не расслышал, повторите")
                    engine.runAndWait()
                except sr.RequestError as e:
                    print("Ошибка сервиса; {0}".format(e))
                    engine.say("Повторите, пожалуйста")
                    engine.runAndWait()
                    print("Повторите, пожалуйста")

        if nameD != "Unknown":
            nameNew=nameD
            for i in known_face_names:
                if i == name_i:
                    seconds = time.time()
                    nameNew = nameD + ";" + str(seconds)
            f = open('base.txt', 'a')
            f.write(nameNew + "\n")
            f.close()


            known_face_names.append(nameNew)
            name2 = nameNew + ".jpg"
            engine.say("Мне нужно вас сфотографировать, разместите лицо по центру экрана и постарайтесь не двигаться.")
            engine.runAndWait()
            print("Мне нужно вас сфотографировать, разместите лицо по центру экрана и постарайтесь не двигаться.")
            for k in range(5, 0, -1):
                i = 0
                while (True):
                    i += 1
                    ret, frame_dial = video_capture.read()
                    cv2.putText(frame_dial, str(k), (280, 280), font, 7.0, (0, 0, 255), 7)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imshow('Video', frame_dial)
                    if i > 40:
                        break
                    # cv2.imshow('frame',gray)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            ret, frame_new = video_capture.read()
            rgb_frame_new = frame_new[:, :, ::-1]
            cv2.imwrite("BasePhoto/Rayan.jpg", rgb_frame_new)
            file_oldname = os.path.join("BasePhoto", "Rayan.jpg")
            file_newname_newfile = os.path.join("BasePhoto", nameNew + ".jpg")
            os.rename(file_oldname, file_newname_newfile)
            #cv2.imwrite("BasePhoto/" + nameNew + ".jpg", rgb_frame_new)
            images.append(face_recognition.load_image_file("BasePhoto/" + nameNew + ".jpg"))
            known_face_encodings.append(face_recognition.face_encodings(images[-1])[0])
            # Добавление в базу выше
            print("Привет, " + nameD + " приятно познакомиться!")
            engine.say("Привет, " + nameD + "приятно познакомиться!")
            engine.runAndWait()
    if not check(name_i):
        print("Приятно было пообщаться, пока!")
        engine.say("Приятно было пообщаться, пока!")
        engine.runAndWait()


def check(TName):
    marker = False
    global process_this_frame
    for q in range(0,3):
        ret, frame = video_capture.read()
        # t1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.cvtColor(t1, cv2.COLOR_GRAY2BGR)

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_names_this = []
        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)


            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names_this.append(name)

        process_this_frame = not process_this_frame
        for pop in face_names_this:
            if pop == TName:
                marker = True
    return marker



# Load a sample picture and learn how to recognize it.

# me_image = face_recognition.load_image_file("me.jpg")
# me_face_encoding = face_recognition.face_encodings(me_image)[0]

# Load a second sample picture and learn how to recognize it.
# biden_image = face_recognition.load_image_file("unknown.jpg")
# biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
# ksusha_image = face_recognition.load_image_file("unknown3.jpg")
# ksusha_face_encoding = face_recognition.face_encodings(ksusha_image)[0]
# Create arrays of known face encodings and their names
#    me_face_encoding,
#   biden_face_encoding,
#   ksusha_face_encoding
# ]

# Initialize some variables

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # t1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.cvtColor(t1, cv2.COLOR_GRAY2BGR)

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)
    if len(face_names):
        dialog()

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
