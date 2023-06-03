
import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.

lokesh_image = face_recognition.load_image_file("lokesh.jpeg")
lokesh_face_encoding = face_recognition.face_encodings(lokesh_image)[0]

sarath_image = face_recognition.load_image_file("sarath.jpeg")
sarath_face_encoding = face_recognition.face_encodings(sarath_image)[0]

bhaskar_image = face_recognition.load_image_file("bhaskar.jpeg")
bhaskar_face_encoding = face_recognition.face_encodings(bhaskar_image)[0]

dinesh_image = face_recognition.load_image_file("dinesh.jpeg")
dinesh_face_encoding = face_recognition.face_encodings(dinesh_image)[0]

feroz_image = face_recognition.load_image_file("feroz.jpeg")
feroz_face_encoding = face_recognition.face_encodings(feroz_image)[0]

kumar_image = face_recognition.load_image_file("kumar.jpeg")
kumar_face_encoding = face_recognition.face_encodings(kumar_image)[0]

aravind_image = face_recognition.load_image_file("aravind.jpg")
aravind_face_encoding = face_recognition.face_encodings(aravind_image)[0]

santoshi_image = face_recognition.load_image_file("santoshi.jpeg")
santoshi_face_encoding = face_recognition.face_encodings(santoshi_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    lokesh_face_encoding,
    sarath_face_encoding,
    bhaskar_face_encoding,
    dinesh_face_encoding,
    feroz_face_encoding,
    kumar_face_encoding,
    aravind_face_encoding,
    santoshi_face_encoding
]
known_face_names = [
    "lokesh",
    "Sarath",
    "bhaskar",
    "dinesh",
    "feroz",
    "kumar",
    "aravind"
    "santoshi"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video here 'ret' means return value of the frame
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing and assign it to a variable called small_frame
    #the code format is cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=2)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.55)
            name = "Unknown"


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
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("Mr.Bhaskar_bachi")
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
