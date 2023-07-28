import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import librosa
from pickle import load
model = load(open('SER_MLP_01_scaled.pkl', 'rb'))
scaler = load(open('scaler.pkl', 'rb'))
def extract_features(y):
  result=np.array([])
  #mfcc
  Mfcc = np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
  result=np.hstack((result,Mfcc))
  #chroma
  stft=np.abs(librosa.stft(y))
  Chroma=np.mean(librosa.feature.chroma_stft(S=stft,sr=sr).T,axis=0)
  result = np.hstack((result,Chroma))
  #mel
  Mel=np.mean(librosa.feature.melspectrogram(y=y,sr=sr).T,axis=0)
  result=np.hstack((result,Mel))
  #zcr
  Zcr =np.mean(librosa.feature.zero_crossing_rate(y=y).T,axis=0)
  result=np.hstack((result,Zcr))
  #rms
  Rms = np.mean(librosa.feature.rms(y=y).T,axis=0)
  result=np.hstack((result,Rms))
  return [result]
def predict_emotion(uploaded_file):
    y, sr = librosa.load(uploaded_file)
    duration=3
    total_duration = librosa.get_duration(y=y,sr=sr)
    num_segments = int(total_duration / duration)
    segments = [[],[]]
    count=[0,0,0,0,0,0]
    for i in range(num_segments):
        start_time = i * duration
        end_time = (i + 1) * duration
        segment = y[int(start_time * sr):int(end_time * sr)]
        x = extract_features(segment)
        x = scaler.transform((np.array(x)).reshape(1,182))
        y_pred = model.predict(np.array(x))
        max_index = np.argmax(y_pred)
        acc=y_pred[0][max_index]*100
        if(max_index == 0):
          y_pred="ANGRY"
          count[0]=count[0]+1
        elif(max_index == 1):
          y_pred="DISGUST"
          count[1]=count[1]+1
        elif(max_index == 2):
          y_pred="FEAR"
          count[2]=count[2]+1
        elif(max_index == 3):
          y_pred="HAPPY"
          count[3]=count[3]+1
        elif(max_index == 4):
          y_pred="NEUTRAL"
          count[4]=count[4]+1
        elif(max_index == 5):
          y_pred="SAD"
          count[5]=count[5]+1
        segments[0].append(y_pred)
        segments[1].append(acc)
    max_emotion=np.argmax(count)
    if(max_emotion == 0):
      em="ANGRY"
    elif(max_emotion == 1):
      em="DISGUST"
    elif(max_emotion == 2):
      em="FEAR"
    elif(max_emotion == 3):
      em="HAPPY"
    elif(max_emotion == 4):
      em="NEUTRAL"
    elif(max_emotion == 5):
      em="SAD"
    return em
def main():
    st.title("Speech Emotion Recognition")
    st.write("Upload an audio file and get the predicted emotion.")

    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

    if st.button("Predict Emotion"):
        if uploaded_file is not None:
            # y, sr = load_audio(uploaded_file)
            predicted_emotion = predict_emotion(uploaded_file)
            st.write("Predicted Emotion:", predicted_emotion)
