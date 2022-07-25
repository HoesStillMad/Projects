from flask import Flask, render_template, request, redirect
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import speech_recognition as sre
from keras.models import load_model


app = Flask(__name__)

@app.route("/", methods = ['GET','POST'])

def index():
    prediction =''
    if request.method =='POST':
        print('Form Data Received')

        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            model_path = 'Accent Recognition/model.h5'
            model = load_model(model_path)

            audiofile = sre.AudioFile(file)
            audio, sample_rate= librosa.load(audiofile, res_type='kaiser_fast')
            mfccs_features= librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccs_scaled_features=np.mean(mfccs_features.T, axis=0)
            
            mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
            
            predicted_label=np.argmax(model.predict(mfccs_scaled_features))
            
            labelencoder=LabelEncoder()
            prediction=labelencoder.inverse_transform(predicted_label.reshape(1))
            
    return render_template('index.html', prediction = prediction)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
