import face_recognition
import pyttsx3
import cv2
import numpy as np
#import pyautogui
import time



# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
known_face_names = []
images = []
known_face_encodings = []

f = open('base.txt', 'r')
known_face_names=f.read().splitlines()
f.close() 
i=0
print(known_face_names)
while i<len(known_face_names):
    images.append(face_recognition.load_image_file("BasePhoto/"+known_face_names[i]+".jpg"))
    known_face_encodings.append(face_recognition.face_encodings(images[i])[0])
    i+=1


# Load a sample picture and learn how to recognize it.

#me_image = face_recognition.load_image_file("me.jpg")
#me_face_encoding = face_recognition.face_encodings(me_image)[0]

# Load a second sample picture and learn how to recognize it.
#biden_image = face_recognition.load_image_file("unknown.jpg")
#biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
#ksusha_image = face_recognition.load_image_file("unknown3.jpg")
#ksusha_face_encoding = face_recognition.face_encodings(ksusha_image)[0]
# Create arrays of known face encodings and their names
#    me_face_encoding,
#   biden_face_encoding,
#   ksusha_face_encoding
#]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
engine = pyttsx3.init()
name="me"
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

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
        left *=4 

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)
    if name=="Unknown":
        engine.say("??, ?? ?????? ???? ????????, ?????? ?????? ??????????")
        engine.runAndWait()
        nameNew = input("?????????????? ??????:")
        i=0
        len_names=len(known_face_names)
        for i in known_face_names:
            if i==nameNew:
                seconds=time.time()
                nameNew=nameNew+"|"+str(seconds)
        f = open('base.txt', 'a')
        f.write(nameNew + "\n")
        f.close()
        known_face_names.append(nameNew)
        name2=nameNew+".jpg"
        engine.say("?????? ?????????? ?????? ????????????????????????????????, ???????????????????? ???????? ???? ???????????? ???????????? ?? ???????????????????????? ???? ??????????????????")
        engine.runAndWait() 
        
        ret, frame_new = video_capture.read()
        rgb_frame_new =frame_new[:, :, ::-1]
        cv2.imwrite("BasePhoto/"+name2, rgb_frame_new)
        images.append(face_recognition.load_image_file("BasePhoto/" + name2))
        known_face_encodings.append(face_recognition.face_encodings(images[-1])[0])



    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()