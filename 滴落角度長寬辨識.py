import cv2
import numpy as np
import os

def detect_black_scale(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 50, 80])

    mask = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_length = 0
    best_rect = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 100:
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
    return None

def detect_red_regions_and_black(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    black_line_length = detect_black_scale(image)
    if black_line_length is None:
        print("Error: No black scale block detected.")
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 偵測亮紅（偏橘紅）
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    # 偵測深紅（你點到的 H=177, V=88 要在這範圍內）
    lower_red2 = np.array([170, 100, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 排序輪廓（左到右）
    sorted_contours = []
    for contour in contours:
        if len(contour) >= 5:
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            sorted_contours.append((contour, cx))

    sorted_contours.sort(key=lambda c: c[1])

    # 計數器初始化
    circle_count = 0

    for contour, cx in sorted_contours:
        ellipse = cv2.fitEllipse(contour)
        (ex, ey), (major_axis, minor_axis), angle = ellipse

        length_cm = major_axis / black_line_length
        width_cm = minor_axis / black_line_length
        if length_cm < 1.1 or width_cm < 1.1:
            continue

        circle_count += 1
        print(f"圓點{circle_count}：{length_cm:.2f} cm x {width_cm:.2f} cm")

        text = f"{length_cm:.2f} cm x {width_cm:.2f} cm"

        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 4)

        text_pos = (cx - text_w // 2, cy - 20)
        cv2.putText(image, text, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 4)

        # ➤ 在橢圓正下方標示「Circle 1、2、3...」
        label = f"Circle {circle_count}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_margin = 20  # 調整這個值控制距離
        label_pos = (cx - lw // 2, int(ey + major_axis / 2) + lh + 70)
        cv2.putText(image, label, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.ellipse(image, ellipse, (255, 0, 0), 3)

    # ➤ 增加空白底部高度
    extra_space = 100
    extended = np.full((image.shape[0] + extra_space, image.shape[1], 3), 255, dtype=np.uint8)
    extended[0:image.shape[0], :] = image

    # ➤ 儲存圖片到桌面
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    output_path = os.path.join(desktop, "output_result.png")
    cv2.imwrite(output_path, extended)
    print("圖片已儲存到：", output_path)

    # ➤ 顯示縮小後的圖片
    scale_percent = 50
    width = int(extended.shape[1] * scale_percent / 100)
    height = int(extended.shape[0] * scale_percent / 100)
    resized = cv2.resize(extended, (width, height))

    cv2.imshow("Detected Red + Black Scale", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用範例（圖片路徑請換成你自己那張）
image_path = r"C:\Users\user\Downloads\40.png"
detect_red_regions_and_black(image_path)

