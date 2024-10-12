import cv2
import numpy as np
import os
import time
import requests

# Fungsi untuk mengunggah gambar ke API
def upload_to_api(image_path, category):
    url = 'http://10.10.123.149/DashboardASV/connection/APIget-post.php'  # Ganti dengan URL API yang sesuai
    
    # Buka gambar sebagai file
    with open(image_path, 'rb') as file:
        files = {'image_data': file}
        data = {'table': category}
        
        # Kirim permintaan POST ke API
        response = requests.post(url, files=files, data=data)
        
        if response.status_code == 200:
            print(f"Gambar berhasil diunggah ke kategori {category}")
        else:
            print(f"Gagal mengunggah gambar: {response.status_code} - {response.text}")

# Fungsi untuk mendeteksi benda kotak berdasarkan warna dengan ukuran minimal
def detect_colored_box(image, min_size, lower_color, upper_color):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_size:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"Position: ({x}, {y}), Area: {area}"
                cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                return True, image
    return False, image

# Membuka dua kamera
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

lower_green = np.array([60, 30, 40])
upper_green = np.array([90, 180, 180])
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])
min_size = 500

output_folder = "output_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

last_capture_time_green = time.time()
last_capture_time_blue = time.time()
capture_delay = 5

while True:
    ret1, frame1 = cap1.read()
    if not ret1:
        print("Tidak dapat membaca dari kamera pertama")
        break
    
    detected_green, result_frame_green = detect_colored_box(frame1, min_size, lower_green, upper_green)
    detected_blue, result_frame_blue = detect_colored_box(frame1, min_size, lower_blue, upper_blue)
    
    if detected_green and (time.time() - last_capture_time_green) > capture_delay:
        image_path_green = os.path.join(output_folder, "kotak_hijau.png")
        cv2.imwrite(image_path_green, result_frame_green)
        print(f"Gambar hijau disimpan di: {image_path_green}")
        
        # Unggah gambar hijau ke API, dengan kategori 'surface'
        upload_to_api(image_path_green, 'surface')
        last_capture_time_green = time.time()

    if detected_blue and (time.time() - last_capture_time_blue) > capture_delay:
        ret2, frame2 = cap2.read()
        if ret2:
            image_path_blue = os.path.join(output_folder, "kotak_biru.png")
            cv2.imwrite(image_path_blue, frame2)
            print(f"Gambar biru disimpan di: {image_path_blue}")
            
            # Unggah gambar biru ke API, dengan kategori 'underwater'
            upload_to_api(image_path_blue, 'underwater')
        last_capture_time_blue = time.time()
    
    cv2.imshow('Green and Blue Detection', result_frame_green)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
