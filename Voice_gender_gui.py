import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import librosa
import joblib
import pandas as pd

# Load the voice gender detection model from .pkl file
model = joblib.load('C:\\Users\\SASIDHAR\\Desktop\\voice_gender\\voice_gender_detection.pkl')

# Initializing the GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Voice Gender Detection')
top.configure(background='#CDCDCD')

# Initializing the Labels
label = Label(top, background="#CDCDCD", font=('arial', 15, "bold"))
file_path_label = Label(top, background="#CDCDCD", font=('arial', 12))
file_path_label.pack()
def feature_extractor(y, sr):
    S = np.abs(librosa.stft(y))

    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tonnetz_mean = np.mean(tonnetz.T, axis=0)
    tonnetz_var = np.var(tonnetz.T, axis=0)
    features = np.append(tonnetz_mean, tonnetz_var)

    spec_centroid = librosa.feature.spectral_centroid(sr=sr, S=S)
    spec_centroid_mean = np.mean(spec_centroid, axis=1)
    spec_centroid_var = np.var(spec_centroid, axis=1)
    features = np.append(features, [spec_centroid_mean, spec_centroid_var])

    mfcc = librosa.feature.mfcc(sr=sr, S=S)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_var = np.var(mfcc.T, axis=0)
    features = np.append(features, mfcc_mean)
    features = np.append(features, mfcc_var)

    spec_width = librosa.feature.spectral_bandwidth(sr=sr, S=S)
    spec_width_mean = np.mean(spec_width)
    spec_width_var = np.var(spec_width)
    features = np.append(features, [spec_width_mean, spec_width_var])

    spec_contrast = librosa.feature.spectral_contrast(sr=sr, S=S)
    spec_contrast_mean = np.mean(spec_contrast.T, axis=0)
    spec_contrast_var = np.var(spec_contrast.T, axis=0)
    features = np.append(features, spec_contrast_mean)
    features = np.append(features, spec_contrast_var)

    return features

# Defining Detect function which detects the gender from the audio file using the model
def Detect(file_path):
    global label


    try:
        audio, sr = librosa.load(file_path, sr=None)

        y = librosa.effects.harmonic(audio)  # Fixed variable name
        features = feature_extractor(y, sr)  # Extract features from audio

        # Reshape features to match the expected input shape for prediction
        features = np.array(features).reshape(1, -1)

        # Predict gender
        gender = model.predict(features)

        label.configure(foreground="#011638", text=f"Predicted Gender: {gender} ")
    except Exception as e:
        label.configure(foreground="#011638", text="Error occurred while processing the audio file.")

# Defining Upload Audio Function
def upload_audio():
    try:
        file_path = filedialog.askopenfilename()
        file_path_label.config(text=f"File Path: {file_path}")
        show_Detect_button(file_path)
    except Exception as e:
        print("Error:", e)

# Defining Show_detect button function
def show_Detect_button(file_path):
    Detect_b = Button(top, text="Detect Gender", command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    Detect_b.place(relx=0.79, rely=0.46)

upload = Button(top, text="Upload an Audio File", command=upload_audio, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)

label.pack(side="bottom", expand=True)
heading = Label(top, text="Voice Gender Detection", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

top.mainloop()
