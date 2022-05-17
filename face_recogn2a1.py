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
from fuzzywuzzy import fuzz

timeout = 10
t = Timer(timeout, os._exit, [1])

nameNew=""
bot_answ=""
count_of_unknowns=0
flag1=False
# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
faceCascade = CascadeClassifier('haa rcascade_frontalface_default.xml')
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
name = "Unknown"
count_of_unknowns = 0
#def tree():

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
        issue="Я тебя не знаю, как тебя зовут?"
        engine.say(issue)
        count_of_unknowns=0
        engine.runAndWait()
        nameD=name_i
        while nameD == "Unknown":
            nameD=speak("Unknown")
        #log(nameD, issue, nameD)
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
            print("Привет, " + nameD + ", приятно познакомиться!")
            engine.say("Привет, " + nameD + ", приятно познакомиться!")
            engine.runAndWait()
    if count_of_unknowns==0:
        question = "Как поживаете?"
        engine.say(question)
        engine.runAndWait()
        answ = speak("000")
        #log(nameD, question, answ, bot_answ)
        #if not check(name_i):
        #    print("Приятно было пообщаться, пока!")
        #    engine.say("Приятно было пообщаться, пока!")
        #    engine.runAndWait()
        question = "Есть ли у вас какие-нибудь вопросы?"
        engine.say(question)
        engine.runAndWait()
        answ = speak("001")
        #log(nameD, question, answ, bot_answ)
        bot_answ=basix(answ)
        engine.say(bot_answ)
        engine.runAndWait()

        #Ответ+реакция
        #Еще кусочек диалога
        question = "Спросите еще что-нибудь?"
        engine.say(question)
        engine.runAndWait()
        answ = speak("002")
        bot_answ = basix(answ)
        engine.say(bot_answ)
        engine.runAndWait()
        #log(nameD, question, answ, bot_answ)
        #Спросить про выставку
        question = "Как нам наша выставка?"
        engine.say(question)
        engine.runAndWait()
        answ = speak("003")
        #log(nameD, question, answ, bot_answ)
        #Предложить посетить стенд телебота
        question = "Очень настоятельно рекомендую вам посетить стенд телебота, там очень интересно!"
        engine.say(question)
        engine.runAndWait()
        answ = speak("004")
        #log(nameD, question, answ, bot_answ)
        question = "Приятно было пообщаться, я поехал, пока!"
        engine.say(question)
        engine.runAndWait()
    #Попрощаться


#Проверка присутствия
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
#функция разговора
def speak(referen):
    with sr.Microphone() as source:
        message = mic.listen(source)
        try:
            referen = mic.recognize_google(message, language="ru-RU")
            print(referen)
        except sr.UnknownValueError:
            print("Я не расслышал, повторите")
            engine.say("Я не расслышал, повторите")
            engine.runAndWait()
        except sr.RequestError as e:
            print("Ошибка сервиса; {0}".format(e))
            engine.say("Повторите, пожалуйста")
            engine.runAndWait()
            print("Повторите, пожалуйста")
    return referen
#Запись логов
def log(nameL, mes, ans, bot_ans):
    seconds = time.time()
    f = open("Logs/Log:: "+nameL+".txt", 'a')

    f.write(seconds +"// " + mes +"\n"+ "user: " + ans + "bot: " + bot_ans + "\n" + "\n")
    f.close()
#функция подсчета строк в базе
def num():
    numberQ = 0
    f = open("QUEST.txt", "r")
    while True:
        line = f.readline()
        if not line:
            break
        numberQ += 1
    f.close()
    return numberQ
#Обращение  к базе
def basix(question):
    match = 0
    match_id = ""
    answ_id = ""
    fqr = open("QUEST.txt", "r")
    while True:
        # считываем строку
        line = fqr.readline()
        if not line:
            break
        line_id = line.split(';')[0]
        line_prase = line.split(';')[1]
        line_prase = line_prase[:-1]
        # matcher = difflib.SequenceMatcher(None, message_text, line_prase)
        matcher = fuzz.WRatio(question, line_prase)

        if matcher > match:
            match = matcher
            match_id = line_id
        if question == line_prase:
            answ_id = line_id
            break
    fqr.close()
    if match > 88:
        print(match)
        answ_id = match_id

    line_ans = ""
    line_id_ans = ""
    if answ_id != "":
        far = open("ANSW.txt", "r")
        while answ_id != line_id_ans:
            line_answ = far.readline()
            if not line_answ:
                break
            line_id_ans = line_answ.split(';')[0]
            line_prase_ans = line_answ.split(';')[1]
            line_prase_ans = line_prase_ans[:-1]
        far.close()
        answer = line_prase_ans
        print(answer)
    else:
        answer="На такой вопрос я ответа не знаю, но я научусь!"
    return answer

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
