import cv2
import numpy as np
import os

def detect_black_scale(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 50, 80]))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = max((cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= 100),
               key=lambda r: max(r[2], r[3]), default=None)
    return max(best[2], best[3]) if best else None

def calculate_true_red_area(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    black_line_length = detect_black_scale(image)
    if black_line_length is None:
        print("Error: No black scale block detected.")
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 80, 50]), np.array([180, 255, 255]))
    mask = mask1 + mask2

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    red_pixel_count = cv2.countNonZero(mask)
    area_cm2 = (red_pixel_count / (black_line_length ** 2)) if black_line_length else 0

    print(f"實際紅色面積：約 {area_cm2:.2f} 平方公分")

    # 顯示紅色區域輪廓方便檢查
    result_img = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)

    # 儲存結果圖檔
    out_path = os.path.join(os.path.dirname(image_path), "true_red_area_marked.png")
    cv2.imwrite(out_path, result_img)
    print(f"紅色標註圖已儲存：{out_path}")

# 請將圖片路徑修改為你自己的圖檔路徑
image_path = r"C:\Users\nicole\Downloads\30.png"
calculate_true_red_area(image_path)

