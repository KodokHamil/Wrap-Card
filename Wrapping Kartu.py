import cv2
import numpy as np

def detect_card(image_path, draw=True):
    # Baca gambar asli
    original = cv2.imread(image_path)
    
    # Buat salinan gambar asli agar gambar asli masih tersimpan
    image = original.copy()
    
    #Konversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Filter greenscreen
    green_lower = np.array([10, 40, 50])  # jika ingin masking warna kuning ubah lower hue = 10, upper hue = 30
    green_upper = np.array([30, 255, 255]) 
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, green_lower, green_upper)
    mask = cv2.bitwise_not(mask)
    greenscreen = cv2.bitwise_and(original, original, mask=mask)
    
    green = greenscreen.copy()
    
    # Deteksi tepi dengan Canny
    edges = cv2.Canny(greenscreen, 50, 150)
    
    # Temukan contour
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_corners = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        # Memeriksa apakah sudutnya 4
        if len(approx) == 4:
            card_corners.append(approx)
            if draw:
                for i in range(len(approx)):
                    start_point = tuple(approx[i][0])
                    end_point = tuple(approx[(i + 1) % len(approx)][0])                  
                    # Gambar tepi kartu
                    cv2.line(green, start_point, end_point, (255, 0, 255), 2)
                    # Lingkari ujung kartu
                    cv2.circle(green, (approx[i][0][0], approx[i][0][1]), 3, (255, 0, 0), 3)

    cv2.imshow('Kartu Asli', original)
    cv2.imshow('masking', greenscreen)
    cv2.imshow('Kartu Terdeteksi', green)
    cv2.waitKey(1)

    return card_corners

def warp(image_path, corners, window_name):
    image = cv2.imread(image_path)
    #sort ini digunakan untuk mengurutkan ujung ujung corner dengan acuan titik x dan y,
    #sehingga titik" dapat terdeteksi kiri atas ke kanan bawah
    corners = sorted(corners, key=lambda x: x[0][0] + x[0][1])  
    pts1 = np.float32([corners[0][0], corners[1][0], corners[2][0], corners[3][0]])
    
    width = max(abs(corners[0][0][0] - corners[1][0][0]), abs(corners[2][0][0] - corners[3][0][0]))
    height = max(abs(corners[0][0][1] - corners[2][0][1]), abs(corners[1][0][1] - corners[3][0][1]))
    
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(image, matrix, (width, height))
    cv2.imshow(window_name, imgOutput)
    cv2.waitKey(1) 

# Deteksi ujung kartu
image_path = 'kartumiring.jpg'
card_corners = detect_card(image_path, draw=True)

# Pastikan ada kartu yang terdeteksi
if card_corners:
    for i, corners in enumerate(card_corners):
        window_name = f'Kartu Wrapped {i+1}'
        warp(image_path, corners, window_name)

cv2.waitKey(0) 
cv2.destroyAllWindows()
