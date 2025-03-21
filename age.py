import tkinter as tk
from tkinter import filedialog
from tkinter import *
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk


# donwload haarcascade_frontalface_default from here "https://github.com/opencv/opencv/tree/master/data/haarcascades"
# Load face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load age detection model
def load_model(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_model("age_model.json", "age_model_weights.h5")

# Age categories (example: modify based on your dataset labels)
AGE_LABELS = ["0-2", "3-6", "7-12", "13-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

# Initialize GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Age Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def detect_age(file_path):
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    if len(faces) == 0:
        label1.configure(foreground="#011638", text="No face detected")
        return

    for (x, y, w, h) in faces:
        face = gray_image[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))  # Resize to model's input size
        face = face.astype("float32") / 255.0  # Normalize
        face = np.expand_dims(face, axis=-1)  # Add channel dimension
        face = np.expand_dims(face, axis=0)   # Add batch dimension

        prediction = model.predict(face)
        predicted_age = AGE_LABELS[np.argmax(prediction)]
        
        print(f"Predicted Age: {predicted_age}")
        label1.configure(foreground="#011638", text=f"Predicted Age: {predicted_age}")

def show_detect_button(file_path):
    detect_b = Button(top, text="Detect Age", command=lambda: detect_age(file_path), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail((top.winfo_width()/2.25, top.winfo_height()/2.25))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')

        show_detect_button(file_path)
    except Exception as e:
        print(f"Error: {e}")

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand=True)
label1.pack(side='bottom', expand=True)
heading = Label(top, text='Age Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

top.mainloop()
