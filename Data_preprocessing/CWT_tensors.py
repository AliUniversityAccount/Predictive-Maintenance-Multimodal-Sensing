import os
import re
import numpy as np
import pandas as pd
import pywt
from skimage.transform import resize

image_folder = r"C:\Users\ALI\Downloads\MVS\HikRobot\Healthy_MV\50percenthealthy"
vib_file = r"C:\Users\ALI\Downloads\MVS\HCHS_DATA_COLLECTION_FINAL\hchs_50_healthy.csv"
save_folder = r"C:\Users\ALI\Downloads\MVS\CWT_TENSORS_50HEALTHY"

os.makedirs(save_folder, exist_ok=True)

df = pd.read_csv(vib_file)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.set_index("Timestamp")

signal_cols = [
    'Joint 2 Acceleration - X',
    'Joint 2 Acceleration - Y',
    'Joint 2 Acceleration - Z',
    'Joint 3 Acceleration - X',
    'Joint 3 Acceleration - Y',
    'Joint 3 Acceleration - Z'
]

fs = 12000

def compute_cwt(signal):
    freqs = np.linspace(50, 3000, 64)
    scales = pywt.central_frequency('cmor1.5-1.0') * fs / freqs

    cwt, _ = pywt.cwt(signal, scales, 'cmor1.5-1.0', sampling_period=1/fs)

    cwt = np.log1p(np.abs(cwt))
    cwt = resize(cwt, (128, 128), mode='reflect', anti_aliasing=True)

    # normalize
    cwt = (cwt - cwt.min()) / (cwt.max() - cwt.min() + 1e-8)

    return cwt

def extract_timestamp(filename):
    match = re.search(r'(\d{14})', filename)
    if match:
        return pd.to_datetime(match.group(1), format="%Y%m%d%H%M%S")
    return None

files = sorted(os.listdir(image_folder))

print("Total images:", len(files))

for file in files:

    if not file.lower().endswith(".jpg"):
        continue

    ts = extract_timestamp(file)

    if ts is None:
        print("No timestamp:", file)
        continue
    start = ts - pd.Timedelta(seconds=2)
    end = ts + pd.Timedelta(seconds=2)

    window = df.loc[start:end]

    if len(window) < 20:
        print("Not enough data:", file)
        continue

    cwt_stack = []

    for col in signal_cols:

        if col not in window.columns:
            print("Missing:", col)
            continue

        sig = window[col].values

        if len(sig) < 10:
            continue

        cwt_map = compute_cwt(sig)
        cwt_stack.append(cwt_map)

    if len(cwt_stack) != 6:
        print("Incomplete channels:", file)
        continue

    cwt_tensor = np.stack(cwt_stack)   # (6,128,128)

    save_path = os.path.join(
        save_folder,
        file.replace(".jpg", ".npy")
    )

    np.save(save_path, cwt_tensor)

    print("Saved:", save_path)
