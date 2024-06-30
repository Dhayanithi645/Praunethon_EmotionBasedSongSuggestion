import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from pytube import YouTube
from pydub import AudioSegment
import io

# Define the list of emotions corresponding to your model's output classes
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Define YouTube links for each emotion
emotion_links = {
    'angry': ["https://youtu.be/RfbehsDXLMo?feature=shared", "https://youtu.be/4AYAcFcFu84?feature=shared",
              "https://youtu.be/jHB8omnU0lA?feature=shared", "https://youtu.be/yXQV6zFDRmg?feature=shared"],
    'calm': ["https://youtu.be/ZWuzH0fW8l0?feature=shared", "https://youtu.be/oNpVIgJrKo8?feature=shared",
             "https://youtu.be/fWajtP80g54?feature=shared"],
    'fearful': ["https://youtu.be/zTejXbV_z-I?feature=shared", "https://youtu.be/AKz8TUrUZN8?feature=shared",
                "https://youtu.be/tkql_yvuSK0?feature=shared"],
    'happy': ["https://youtu.be/S9f13Cw2t0M?feature=shared", "https://youtu.be/gHFW5JVGE4U?feature=shared",
              "https://youtu.be/V7dQAzPxw2g?feature=shared"],
    'neutral': ['https://youtu.be/ZWuzH0fW8l0?feature=shared', "https://youtu.be/oNpVIgJrKo8?feature=shared"],
    'sad': ["https://youtu.be/0mCLw42xVYs?feature=shared", "https://youtu.be/TDrChzXWX3s?feature=shared",
            "https://youtu.be/SO4Z1bxjsWA?feature=shared"],
    'surprised': ["https://youtu.be/pXqdAaMi6VY?feature=shared", "https://youtu.be/IjuSj6mlCJo?feature=shared",
                  "https://youtu.be/TLKu73YX7Sk?feature=shared"]
}


# Function to extract features from audio
def extract_features(data, sample_rate):
    result = np.array([])

    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result


# Function to predict emotion from audio
def predict_emotion(audio_path, model_path):
    data, sample_rate = librosa.load(audio_path, sr=None)
    features = extract_features(data, sample_rate)
    features = np.expand_dims(features, axis=0)

    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(features)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_emotion = emotion_labels[predicted_index]

    return predicted_emotion


# Predefined model path
MODEL_PATH = 'SER_model.h5'

# Streamlit UI
st.title("Emotion Recognition from Audio")
st.write("Upload an audio file to predict the emotion")

audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "ogg"])

if audio_file is not None:
    with st.spinner('Processing...'):
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_file.getbuffer())

        predicted_emotion = predict_emotion("temp_audio.wav", MODEL_PATH)

        # Randomly select a YouTube link based on predicted emotion
        if predicted_emotion in emotion_links:
            youtube_link = np.random.choice(emotion_links[predicted_emotion])
            st.write(f"Predicted Emotion: {predicted_emotion}")
            st.write(f"Selected YouTube link: {youtube_link}")

            try:
                # Download and play audio
                yt = YouTube(youtube_link)
                audio_stream = yt.streams.filter(only_audio=True).first()
                audio_bytes = io.BytesIO()
                audio_stream.stream_to_buffer(audio_bytes)
                st.audio(audio_bytes, format='audio/mp3')
            except Exception as e:
                st.error(f"Error playing audio: {str(e)}")

else:
    st.write("Please upload an audio file.")
