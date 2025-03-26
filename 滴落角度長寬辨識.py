import cv2
import numpy as np

def detect_black_scale(image):
    """從黑色色塊中找出長邊最長者作為 1 公分比例尺，並在圖上標記"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 50, 80])  # 放寬亮度容忍範圍

    mask = cv2.inRange(hsv, lower_black, upper_black)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_length = 0
    best_rect = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 100:
            continue

        length = max(w, h)
        if length > max_length:
            max_length = length
            best_rect = (x, y, w, h)

    if best_rect:
        x, y, w, h = best_rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.putText(image, "1 cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
        return max(w, h)
    else:
        return None

def detect_red_regions_and_black(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # 取得黑色比例尺長度（像素）
    black_line_length = detect_black_scale(image)
    if black_line_length is None:
        print("Error: No black scale block detected.")
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 紅色範圍遮罩
    lower_red1 = np.array([0, 50, 154])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 50, 154])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (x, y), (major_axis, minor_axis), angle = ellipse

            # 換算為公分
            length_cm = major_axis / black_line_length
            width_cm = minor_axis / black_line_length

            if length_cm < 1.1 or width_cm < 1.1:
                continue

            text = f"{length_cm:.2f} cm x {width_cm:.2f} cm"

            # 偏移顯示座標，避免文字重疊
            offset_x = 30
            offset_y = -30 - (i % 3) * 20

            cv2.putText(image, text, (int(x + offset_x), int(y + offset_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 4)
            cv2.ellipse(image, ellipse, (255, 0, 0), 3)

    # 顯示圖片（縮小一半）
    scale_percent = 50
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized = cv2.resize(image, (width, height))

    cv2.imshow("Detected Red + Black Scale", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用範例
image_path = r"C:\Users\nicole\Downloads\20.png"
detect_red_regions_and_black(image_path)

