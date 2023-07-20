import streamlit as st
import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment
from keras.models import load_model
from PIL import Image

# Function to extract audio features
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Load the trained model
model = load_model('your_model.h5')

# Function to classify speech emotion
def classify_emotion(audio_path):
    # Load audio file
    audio = AudioSegment.from_file(audio_path)
    audio.export("temp.wav", format="wav")

    # Extract audio features
    features = extract_mfcc("temp.wav")
    features = np.expand_dims(features, 0)
    features = np.expand_dims(features, -1)

    # Classify emotion using the loaded model
    prediction = model.predict(features)
    emotion_label = np.argmax(prediction)
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
    emotion = emotions[emotion_label]

    return emotion

# Main streamlit app
def main():
    st.title("Speech Emotion Recognition")
    st.write("Upload an audio file to classify the emotion.")

    # File uploader
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav'])

    # Display file info and classify emotion on button click
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        if st.button("Classify Emotion"):
            audio_path = "temp_audio.wav"
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            emotion = classify_emotion(audio_path)

            st.write("Predicted Emotion: ", emotion)

            # Display emoji and background image based on emotion
            emoji = get_emoji(emotion)
            st.write(emoji)


    st.text("")  # Empty space for formatting

def get_emoji(emotion):
    emoji_mapping = {
        'angry': '😠',
        'disgust': '🤢',
        'fear': '😨',
        'happy': '😄',
        'neutral': '😐',
        'ps': '😶',
        'sad': '😢'
    }
    return emoji_mapping.get(emotion, '')

# Run the app
if __name__ == "__main__":
    main()
