import os
import pickle

DATA_FILE = r"C:\Users\t4kic\Documents\JapanHorseRacePrediction\data\models\deepfm\deepfm_data.pkl"
OUTPUT_FILE = r"C:\Users\t4kic\Documents\JapanHorseRacePrediction\models\deepfm\deepfm_metadata.pkl"

def extract_metadata():
    if not os.path.exists(DATA_FILE):
        print(f"File not found: {DATA_FILE}")
        return

    print("Loading large pickle file...")
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)

    metadata = {
        'feature_config': data['feature_config'],
        'label_encoders': data['label_encoders']
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"Extracted metadata to {OUTPUT_FILE}")
    print(f"Size: {os.path.getsize(OUTPUT_FILE) / 1024:.2f} KB")

if __name__ == "__main__":
    extract_metadata()
