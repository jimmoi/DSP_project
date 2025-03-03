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
    import queue
    import time
    
def list_audio_devices():
        """ Lists all available audio input devices """
        print("Available audio input devices:")
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                print(f"{i}: {d['name']}")

def audio_to_melspectrogram(audio, sr=22050):
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB
    return mel_spec_db

def classify_live_audio(model, map_index_class, input_device, window_size=0.5, overlap=0.25, sr=22050):
    """ 
    Real-time audio classification using overlapping windows. 
    Example: 
        window_size = 0.5 sec, overlap = 0.25 sec → new prediction every 0.25 sec.
    """
    
    # Compute samples per window
    samples_per_window = int(window_size * sr)
    step_size = int((1 - overlap) * samples_per_window)  # Step size based on overlap

    # Set audio input device
    sd.default.device = input_device
    buffer = deque(maxlen=samples_per_window)  # Circular buffer for overlapping samples

    # Initialize plot
    plt.ion()
    fig, ax = plt.subplots()
    _ = input("press any button to start")
    class_labels = list(map_index_class.values())

    print(f"Listening on device {sd.query_devices(sd.default.device)['name']}...")
    
    # Queue to store results from callback
    prediction_queue = queue.Queue()
    
    def audio_callback(indata, frames, time, status):
        """ Callback function for real-time audio input """
        if status:
            print(f"Error: {status}")
        buffer.extend(indata[:, 0])  # Add new audio samples to buffer
        
        if len(buffer) >= samples_per_window:  # Only predict when buffer is filled
            audio = np.array(buffer)
            audio = librosa.util.normalize(audio)
            mel_spec = audio_to_melspectrogram(audio, sr=sr)
            mel_spec = torch.tensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)

            # Get sorted predictions
            with torch.no_grad():
                output = model(mel_spec)
                probabilities = F.softmax(output, dim=1).cpu().numpy().flatten()
            sorted_indices = np.argsort(probabilities)
            sorted_probs = probabilities[sorted_indices][-5:]
            sorted_labels = [class_labels[i] for i in sorted_indices][-5:]
            prediction_queue.put([sorted_probs, sorted_labels])

    # Start real-time audio stream
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", callback=audio_callback):
        while True:
            try:
                # Process the latest prediction from the queue
                result = prediction_queue.get(timeout=0.1)
                if result:
                    sorted_probs_ = result[0]
                    sorted_labels_ = result[1]
                    # Update plot (Main Thread)
                    ax.clear()
                    ax.barh(sorted_labels_, sorted_probs_, color="blue")
                    ax.set_xlim(0, 1)
                    ax.set_xlabel("Probability")
                    ax.set_title("Real-Time Classification")
                    plt.draw()
                    plt.pause(0.01)

                    # Print top prediction
                    predicted_class = sorted_labels_[-1]
                    print(f"Predicted Instrument: {predicted_class} ({sorted_probs_[-1]:.2f})")

            except queue.Empty:
                pass  # No new predictions, continue loop




if __name__ == "__main__":
    
    ## load model
    with timer("load model's weight"):
        device = "cuda" if torch.cuda.is_available else "cpu"
        model = InstrumentClassifier_CBAM(28)
        model.to(device)
        model.load_state_dict(torch.load(r'H:\DSP_project\ignoredir\model\instrument_classifier_cbam_300_0.00001_best2.pt'))
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
    
    
    classify_live_audio(model=model, map_index_class=map_index_class, input_device=device_index, sr=22050, window_size=1)
    
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
    

