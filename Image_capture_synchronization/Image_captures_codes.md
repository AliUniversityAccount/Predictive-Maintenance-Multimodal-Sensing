import time
import os
import paramiko
import cv2

SAVE_FOLDER = r"C:\Users\ALI\Downloads\MVS\RaspberryPiCamera\Unhealthy_Pi"
os.makedirs(SAVE_FOLDER, exist_ok=True)

TRIGGER_FILE = r"C:\vision_sync\trigger.txt"
os.makedirs(os.path.dirname(TRIGGER_FILE), exist_ok=True)

PI_HOST = "raspberrypi.local"
PI_USER = "mypi"
PI_PASSWORD = "2005"

COOLDOWN_SECONDS = 2
MAX_IMAGES = 204
image_count = 0

# Preview image command
RPICAM_PREVIEW = (
    "rpicam-still --nopreview --timeout 800 "
    "--width 640 --height 480 -o {path}"
)

# Final high resolution image
RPICAM_FINAL = (
    "rpicam-still --nopreview --timeout 2000 "
    "--width 4608 --height 2592 "
    "--quality 100 "
    "--sharpness 1.0 "
    "--contrast 1.2 "
    "--shutter 5000 "
    "-o {path}"
)

DICTIONARIES = [
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_4X4_100,
    cv2.aruco.DICT_4X4_250,
    cv2.aruco.DICT_4X4_1000,
    cv2.aruco.DICT_5X5_50,
    cv2.aruco.DICT_5X5_100,
    cv2.aruco.DICT_5X5_250,
    cv2.aruco.DICT_5X5_1000,
    cv2.aruco.DICT_6X6_50,
    cv2.aruco.DICT_6X6_100,
    cv2.aruco.DICT_6X6_250,
    cv2.aruco.DICT_6X6_1000,
    cv2.aruco.DICT_7X7_50,
    cv2.aruco.DICT_7X7_100,
    cv2.aruco.DICT_7X7_250,
    cv2.aruco.DICT_7X7_1000
]

aruco_params = cv2.aruco.DetectorParameters()


def wait_for_file_ready(sftp, path, timeout=5):
    start = time.time()

    while True:
        try:
            attr = sftp.stat(path)
            if attr.st_size > 0:
                return True
        except FileNotFoundError:
            pass

        if time.time() - start > timeout:
            return False

        time.sleep(0.1)


def detect_any_marker(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for dict_id in DICTIONARIES:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            print(f"Marker detected using dictionary {dict_id}")
            return True

    return False


def remote_file_exists(sftp, path):
    try:
        sftp.stat(path)
        return True
    except:
        return False


def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return

    rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    h, w = rotated.shape[:2]
    s = min(h, w)

    y1 = (h - s) // 2
    y2 = y1 + s

    cropped = rotated[y1:y2, 0:w]

    zoom = int(s * 0.2 / 2)
    zoomed = cropped[zoom:s-zoom, zoom:s-zoom]

    zoomed = cv2.resize(zoomed, (s, s))

    cv2.imwrite(image_path, zoomed)


print("Looking for marker")

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect(PI_HOST, username=PI_USER, password=PI_PASSWORD)

sftp = ssh.open_sftp()

print("Connected to Pi")

remote_preview = f"/home/{PI_USER}/preview.jpg"

try:
    while True:

        if image_count >= MAX_IMAGES:
            print(f"\nReached {MAX_IMAGES} images. Stopping program.")
            break

        # Capture preview
        stdin, stdout, stderr = ssh.exec_command(
            RPICAM_PREVIEW.format(path=remote_preview)
        )
        stdout.channel.recv_exit_status()

        if not wait_for_file_ready(sftp, remote_preview, timeout=5):
            print("Preview file not ready, skipping")
            continue

        local_preview = os.path.join(SAVE_FOLDER, "preview.jpg")
        sftp.get(remote_preview, local_preview)

        # Detect marker
        if detect_any_marker(local_preview):

            timestamp = time.strftime("%Y%m%d_%H%M%S")

            remote_final = f"/home/{PI_USER}/pi_{timestamp}.jpg"
            local_final = os.path.join(SAVE_FOLDER, f"pi_{timestamp}.jpg")

            print(f"Marker detected! Timestamp: {timestamp}")

            
            print("Waiting 2 seconds before trigger")
            time.sleep(1)

            # Trigger MV camera
            tmp_file = TRIGGER_FILE + ".tmp"

            with open(tmp_file, "w") as f:
                f.write(timestamp)

            os.replace(tmp_file, TRIGGER_FILE)

            print("MV trigger sent")

            # Capture Pi image
            print("Starting Pi capture")

            stdin, stdout, stderr = ssh.exec_command(
                RPICAM_FINAL.format(path=remote_final)
            )
            stdout.channel.recv_exit_status()

            if wait_for_file_ready(sftp, remote_final, timeout=5):
                sftp.get(remote_final, local_final)

                preprocess_image(local_final)

                print(f"Pi image saved: {local_final}")

                image_count += 1
                print(f"Captured {image_count}/{MAX_IMAGES}")

            else:
                print("Final image not ready")

            print(f"Cooldown {COOLDOWN_SECONDS}s\n")
            time.sleep(COOLDOWN_SECONDS)

except KeyboardInterrupt:
    print("\nStopped by user!")

finally:
    sftp.close()
    ssh.close()
    print("Connection closed")


THE BELOW IS THE CODE FOR THE IMAGE SERVER 

import subprocess

subprocess.Popen(["python", "PI_CODENEWNEW.py"])
subprocess.Popen(["python", "mv_codeNEW.py"])

print("Pi & MV synchronized capture started")


THE BELOW IS THE CODE FOR THE HIGH-COST CAMERA

import os
import time
import pyautogui

pyautogui.FAILSAFE = False
SAVE_BTN = (404, 128)
TRIGGER_FILE = r"C:\vision_sync\trigger.txt"

print("MV camera waiting for Pi trigger")

last_trigger = None

while True:
    if not os.path.exists(TRIGGER_FILE):
        time.sleep(0.01)
        continue

    with open(TRIGGER_FILE, "r") as f:
        trigger = f.read().strip()

    if trigger != last_trigger:
        time.sleep(0.01)  # small delay to ensure complete write
        pyautogui.click(SAVE_BTN)
        print(f"MV capture triggered! ({trigger})")
        last_trigger = trigger
        os.remove(TRIGGER_FILE)

    time.sleep(0.01)
