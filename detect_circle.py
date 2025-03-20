import cv2
import numpy as np

# 讀取影像
image = cv2.imread(r"C:\Users\user\Downloads\60.png")

# 轉換為灰階
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 找黑色比例尺 (黑色在灰階接近 0)
_, scale_binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)

# 找比例尺輪廓
scale_contours, _ = cv2.findContours(scale_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 確保有找到比例尺
if not scale_contours:
    print("❌ 未找到比例尺，請檢查圖片")
    exit()

# 假設最長的黑色輪廓是比例尺
scale_contour = max(scale_contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(scale_contour)

# 計算 1 公分對應的像素
scale_pixels = max(w, h)
cm_to_px = scale_pixels / 1.0  # 1 公分的像素數
print(f"✅ 1 公分 ≈ {cm_to_px:.2f} 像素")

# 轉換為 HSV 以偵測紅色血滴
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])

# 合併紅色範圍
red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = red_mask1 + red_mask2

# 去除雜訊
kernel = np.ones((5, 5), np.uint8)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

# 找血滴輪廓
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

valid_drops = []

# 遍歷所有血滴
for cnt in contours:
    if len(cnt) >= 5:
        # **擬合橢圓並獲取長軸與短軸**
        ellipse = cv2.fitEllipse(cnt)
        (center_x, center_y), (major_axis_px, minor_axis_px), angle = ellipse

        # **轉換長軸與短軸為公分**
        major_axis_cm = major_axis_px / cm_to_px
        minor_axis_cm = minor_axis_px / cm_to_px

        # **計算長軸與短軸的端點**
        angle_rad = np.deg2rad(angle)
        dx_major = (major_axis_px / 2) * np.cos(angle_rad)
        dy_major = (major_axis_px / 2) * np.sin(angle_rad)
        dx_minor = (minor_axis_px / 2) * np.sin(angle_rad)
        dy_minor = (minor_axis_px / 2) * np.cos(angle_rad)

        # 長軸兩端點
        major_pt1 = (int(center_x - dx_major), int(center_y - dy_major))
        major_pt2 = (int(center_x + dx_major), int(center_y + dy_major))

        # 短軸兩端點
        minor_pt1 = (int(center_x - dx_minor), int(center_y + dy_minor))
        minor_pt2 = (int(center_x + dx_minor), int(center_y - dy_minor))

        # **繪製橢圓 (藍色)**
        cv2.ellipse(image, ellipse, (255, 0, 0), 2)

        # **繪製長軸與短軸的直線**
        cv2.line(image, major_pt1, major_pt2, (0, 255, 0), 2)  # 長軸 (綠色)
        cv2.line(image, minor_pt1, minor_pt2, (255, 255, 0), 2)  # 短軸 (黃色)

        # **標記長軸與短軸的長度**
        text_x_major = int((major_pt1[0] + major_pt2[0]) / 2)
        text_y_major = int((major_pt1[1] + major_pt2[1]) / 2 - 10)

        text_x_minor = int((minor_pt1[0] + minor_pt2[0]) / 2)
        text_y_minor = int((minor_pt1[1] + minor_pt2[1]) / 2 + 10)

        cv2.putText(image, f"{major_axis_cm:.2f} cm", (text_x_major, text_y_major),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(image, f"{minor_axis_cm:.2f} cm", (text_x_minor, text_y_minor),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3, cv2.LINE_AA)

        # 儲存結果
        valid_drops.append({
            "中心座標 (px)": (int(center_x), int(center_y)),
            "長軸 (cm)": major_axis_cm,
            "短軸 (cm)": minor_axis_cm,
            "旋轉角度 (°)": angle
        })

# 顯示影像
cv2.imshow('Blood Drop Analysis', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 印出結果
if valid_drops:
    print("\n📊 血滴分析結果 (單位: cm)：")
    for idx, drop in enumerate(valid_drops):
        print(f"\n血滴 {idx+1}:")
        print(f"🔸 中心座標: {drop['中心座標 (px)']}")
        print(f"🔸 長軸: {drop['長軸 (cm)']:.2f} cm")
        print(f"🔸 短軸: {drop['短軸 (cm)']:.2f} cm")
        print(f"🔸 旋轉角度: {drop['旋轉角度 (°)']:.2f}°")
else:
    print("❌ 沒有符合條件的血滴")

