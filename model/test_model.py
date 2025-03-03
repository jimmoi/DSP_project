import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
from my_utils import timer

with timer("import lib"):
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    import torch.nn.functional as F
    import librosa
    import pickle
    import sounddevice as sd
    from train_model.model import InstrumentClassifier_CBAM
    from collections import deque
    
def list_audio_devices():
        """ Lists all available audio input devices """
        print("Available audio input devices:")
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                print(f"{i}: {d['name']}")

def audio_to_melspectrogram(audio, sr=22050):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB
    return mel_spec_db

def predictions(sound, model, device, onehot, map_index_class):
    """need window_size = 0.5 second, and sample rate 22050
    so sample = 11025"""
    
    #transform 
    y = librosa.util.normalize(sound)
    y = audio_to_melspectrogram(y)
    y = torch.tensor(y).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        y = y.to(device)
        # Get raw logits from the model
        outputs = model(y)  # Raw logits
        # Apply softmax to the logits to get probabilities
        probabilities = F.softmax(outputs, dim=1)
        probabilities = probabilities.cpu().numpy()[0]
    
    result = {map_index_class[i]: probabilities[i]*100 for i in range(len(probabilities))}
    
    return result

def classify_live_audio(model, map_index_class, input_device, window_size=0.5, overlap=0.25, sr=22050):
    """ Real-time audio classification with loopback support and probability plot """
    # Select input device
    if input_device is not None:
        sd.default.device = input_device

    # Initialize plot
    plt.ion()
    fig, ax = plt.subplots()
    _ = input("press any button to start")
    bars = None
    class_labels = list(map_index_class.values())  # Assuming model has `fc` layer

    print(f"Listening on device {sd.query_devices(sd.default.device)['name']}...")


    while True:
        audio = sd.rec(int(window_size * sr), samplerate=sr, channels=1, dtype="float32", device=input_device)
        sd.wait()
        audio = audio.flatten()

        # Convert audio to mel spectrogram
        audio = librosa.util.normalize(audio)
        mel_spec = audio_to_melspectrogram(audio, sr=sr)
        mel_spec = torch.tensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model(mel_spec)
            probabilities = F.softmax(output, dim=1).cpu().numpy().flatten()

        # Sort probabilities
        sorted_indices = np.argsort(probabilities)
        sorted_probs = probabilities[sorted_indices]
        sorted_labels = [class_labels[i] for i in sorted_indices]

        # Update plot
        ax.clear()
        ax.barh(sorted_labels, sorted_probs, color="blue")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Real-Time Classification Probabilities")
        plt.draw()
        plt.pause(1/60)

        # Print top prediction
        predicted_class = sorted_labels[-1]
        print(f"Predicted Instrument: {predicted_class} ({sorted_probs[-1]:.2f})")

if __name__ == "__main__":
    
    ## load model
    with timer("load model's weight"):
        device = "cuda" if torch.cuda.is_available else "cpu"
        model = InstrumentClassifier_CBAM(28)
        model.to(device)
        model.load_state_dict(torch.load(r'D:\git_project\DSP_project\model\instrument_classifier_cbam_200_0.0001_weight.pt'))
        model.eval()
        
        
    with open(r"model\train_model\encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    onehot = encoder["encoder"]
    map_index_class = encoder["map_index_class"]
    
    with timer("list audio devices"):
        list_audio_devices()
    
    device_index = input("select your divice: ")
    if not device_index.isnumeric():
        raise TypeError("please enter integer number in device list")
    else:
        device_index = int(device_index)
    
    
    classify_live_audio(model=model, map_index_class=map_index_class, input_device=device_index, sr=22050, window_size=0.5)
    
    # ## input 
    # file_path = r"H:\DSP_project\ignoredir\dataset\archive2\Train_submission\Train_submission\violin_sound (241).wav"
    # sr = 22050
    # t = 0.5
    # y, sr = librosa.load(file_path, sr=sr)
    # test_input = y[:int(t*sr)]

    # ## predict
    # with timer("prediction"):
    #     res = predictions(sound=test_input,
    #                 model=model,
    #                 device=device,
    #                 onehot=onehot,
    #                 map_index_class=map_index_class)

    # ## visualize
    # x = res.keys()
    # y = res.values()
    # plt.figure(figsize=(20,10))
    # plt.xticks(rotation=45)
    # plt.bar(x,y)
    # plt.show()
    

