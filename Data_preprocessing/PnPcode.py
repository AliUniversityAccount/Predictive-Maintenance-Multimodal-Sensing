import cv2
import numpy as np
import os
import pandas as pd

image_folder = r"C:\Users\ALI\Downloads\MVS\HikRobot\Healthy_MV\50percenthealthy"
output_excel = r"C:\Users\ALI\Downloads\MVS\pnp_results_mvHEALTHY.xlsx"

camera_matrix = np.array([
    [1000, 0, 640],
    [0, 1000, 360],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((5, 1))

marker_length = 0.05  # meters

aruco_dicts = [
    cv2.aruco.DICT_4X4_50, cv2.aruco.DICT_4X4_100,
    cv2.aruco.DICT_4X4_250, cv2.aruco.DICT_4X4_1000,
    cv2.aruco.DICT_5X5_50, cv2.aruco.DICT_5X5_100,
    cv2.aruco.DICT_5X5_250, cv2.aruco.DICT_5X5_1000,
    cv2.aruco.DICT_6X6_50, cv2.aruco.DICT_6X6_100,
    cv2.aruco.DICT_6X6_250, cv2.aruco.DICT_6X6_1000,
    cv2.aruco.DICT_7X7_50, cv2.aruco.DICT_7X7_100,
    cv2.aruco.DICT_7X7_250, cv2.aruco.DICT_7X7_1000,
    cv2.aruco.DICT_ARUCO_ORIGINAL
]

def get_detector_params():
    params = cv2.aruco.DetectorParameters()

    # Relax detection (important for blur)
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 50
    params.adaptiveThreshWinSizeStep = 5

    params.minMarkerPerimeterRate = 0.01
    params.maxMarkerPerimeterRate = 4.0

    params.polygonalApproxAccuracyRate = 0.1
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    return params

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try multiple enhancements
    processed_versions = []

    # 1. CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    processed_versions.append(clahe.apply(gray))

    # 2. Sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    processed_versions.append(sharpened)

    # 3. Adaptive threshold
    thresh = cv2.adaptiveThreshold(gray,255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,11,2)
    processed_versions.append(thresh)

    blur = cv2.GaussianBlur(gray, (5,5), 0)
    sharpen_blur = cv2.filter2D(blur, -1, kernel)
    processed_versions.append(sharpen_blur)

    return processed_versions

def detect_pose(image):
    params = get_detector_params()

    # -------- FIRST: NORMAL DETECTION --------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for dict_type in aruco_dicts:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_length, camera_matrix, dist_coeffs
            )
            return tvecs[0][0], rvecs[0][0]

    # -------- SECOND: PREPROCESS ONLY IF FAILED --------
    processed_imgs = preprocess_image(image)

    for proc in processed_imgs:
        for dict_type in aruco_dicts:
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)

            corners, ids, _ = detector.detectMarkers(proc)

            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_length, camera_matrix, dist_coeffs
                )
                return tvecs[0][0], rvecs[0][0]

    return None, None

# MAIN
results = []

files = sorted(os.listdir(image_folder))

for file in files:
    if not file.lower().endswith(('.jpg','.png','.jpeg')):
        continue

    path = os.path.join(image_folder, file)
    img = cv2.imread(path)

    if img is None:
        print(f"Cannot read: {file}")
        continue

    tvec, rvec = detect_pose(img)

    if tvec is not None:
        print(f"{file}")
        print(f"   Tvec: {tvec}")
        print(f"   Rvec: {rvec}")

        results.append({
            "image": file,
            "tx": tvec[0],
            "ty": tvec[1],
            "tz": tvec[2],
            "rx": rvec[0],
            "ry": rvec[1],
            "rz": rvec[2]
        })
    else:
        print(f"No marker detected: {file}")

# ==============================
# SAVE RESULTS
# ==============================
df = pd.DataFrame(results)
df.to_excel(output_excel, index=False)

print("\n🎉 DONE - Saved to:", output_excel)
