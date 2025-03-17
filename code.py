import cv2
import numpy as np
import os

def detect_red_dots(image, scale):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("未找到紅點")
        return None, None

    red_dots = []
    for contour in contours:
        if cv2.contourArea(contour) > 20:
            x, y, w, h = cv2.boundingRect(contour)
            real_width_cm = w * scale
            real_height_cm = h * scale
            if real_width_cm >= 0.9 and real_height_cm >= 0.9:  # 忽略長寬小於 0.9 公分的紅點
                red_dots.append((x, y, w, h))

    return red_dots, red_mask


def detect_blue_line(image):
    # 轉換為 HSV 色彩空間
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 設定藍色的 HSV 範圍
    lower_blue = np.array([90, 50, 50])   # 放寬 HSV 閾值
    upper_blue = np.array([150, 255, 255])

    # 建立遮罩
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 形態學處理：閉運算（消除小黑點）、開運算（消除小白點）
    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

    # 找出輪廓
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("未找到藍色區域")
        return None

    # 找出面積最大的輪廓
    max_contour = max(contours, key=cv2.contourArea)
    
    # 取得最小外接矩形（可以是旋轉的）
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)  # 取得矩形 4 個角點
    box = np.int0(box)  # 轉換為整數

    # 計算矩形的長邊
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = box
    length1 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    length2 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)

    # 取最長的邊作為藍線長度
    if length1 > length2:
        x_start, y_start = x1, y1
        x_end, y_end = x2, y2
    else:
        x_start, y_start = x2, y2
        x_end, y_end = x3, y3

    # 確保它是一個細長形（長寬比大於 1.8）
    aspect_ratio = max(length1, length2) / min(length1, length2) if min(length1, length2) != 0 else 0
    if aspect_ratio < 1.8:
        print("偵測到的藍色區域不是長條形")
        return None

    return (x_start, y_start, x_end, y_end)

def calculate_scale(blue_line_points):
    real_length_cm = 1.0
    x1, y1, x2, y2 = blue_line_points
    pixel_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    scale = real_length_cm / pixel_length
    print(f"藍線像素長度: {pixel_length:.2f}, 比例尺: {scale:.6f} cm/像素")
    return scale

def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("無法讀取圖片")
        return

    # 檢測藍線
    blue_line_points = detect_blue_line(image)
    if blue_line_points is None:
        return

    # 計算比例尺
    scale = calculate_scale(blue_line_points)
    if scale is None:
        return

    # 檢測紅點（加入比例尺參數）
    red_dots, red_mask = detect_red_dots(image, scale)
    if red_dots is None:
        return

    # 標示紅點
    for i, (x, y, w, h) in enumerate(red_dots):
        real_width_cm = w * scale
        real_height_cm = h * scale
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"Dot {i+1}: {real_width_cm:.2f}x{real_height_cm:.2f} cm"
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 0), 3)

    # 標示藍線
    x1, y1, x2, y2 = blue_line_points
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(image, "1 cm", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 0, 0), 3)

    # 縮小顯示尺寸
    scale_percent = 25
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(red_mask, (width, height), interpolation=cv2.INTER_AREA)

    # 顯示結果
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Red Mask", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", resized_image)
    cv2.imshow("Red Mask", resized_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 儲存結果到桌面
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    output_path = os.path.join(desktop_path, "result.png")
    cv2.imwrite(output_path, image)
    print(f"結果已儲存至: {output_path}")

    # 輸出結果
    for i, (x, y, w, h) in enumerate(red_dots):
        real_width_cm = w * scale
        real_height_cm = h * scale
        print(f"紅點 {i+1}: 寬度 = {real_width_cm:.2f} 公分, 高度 = {real_height_cm:.2f} 公分")

if __name__ == "__main__":
    image_path = r"(圖片路徑)"
    main(image_path)

