import cv2
import numpy as np

# 讀取影像
image = cv2.imread(r"C:\Users\user\Downloads\60.png")

# 轉換為灰階
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 找黑色比例尺 (黑色在灰階接近 0)
_, scale_binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
scale_contours, _ = cv2.findContours(scale_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if not scale_contours:
    print("❌ 未找到比例尺，請檢查圖片")
    exit()

scale_contour = max(scale_contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(scale_contour)
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

# **擴展血滴範圍，確保邊界完整**
kernel = np.ones((7, 7), np.uint8)  # 擴大 7x7 區域
red_mask = cv2.dilate(red_mask, kernel, iterations=1)

# 找血滴輪廓 (不忽略邊界點)
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

valid_drops = []

for cnt in contours:
    if len(cnt) >= 5:
        # **計算最小內接圓**
        (circle_x, circle_y), circle_radius = cv2.minEnclosingCircle(cnt)
        circle_diameter_cm = (2 * circle_radius) / cm_to_px

        # **忽略直徑小於 1 公分的圓點**
        if circle_diameter_cm < 1:
            continue

        # **使用凸包確保輪廓完整**
        hull = cv2.convexHull(cnt)

        # **確保擬合橢圓時邊界完整**
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon, True)
        ellipse = cv2.fitEllipse(cnt)
        (center_x, center_y), (major_axis_px, minor_axis_px), angle = ellipse

        # **計算最小外接矩形，作為參考**
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        rect_width, rect_height = rect[1]

        # **比較橢圓與矩形長軸，確保更準確**
        major_axis_px = max(major_axis_px, rect_width, rect_height)
        minor_axis_px = min(minor_axis_px, rect_width, rect_height)

        # **如果角度差異過大，使用 minAreaRect() 的角度**
        if abs(angle - rect[2]) > 30:
            angle = rect[2]

        # **轉換長軸與短軸為公分**
        major_axis_cm = major_axis_px / cm_to_px
        minor_axis_cm = minor_axis_px / cm_to_px

        # **計算長軸與短軸的端點**
        angle_rad = np.deg2rad(angle)
        dx_major = (major_axis_px / 2) * np.cos(angle_rad)
        dy_major = (major_axis_px / 2) * np.sin(angle_rad)
        dx_minor = (minor_axis_px / 2) * np.sin(angle_rad)
        dy_minor = (minor_axis_px / 2) * np.cos(angle_rad)

        major_pt1 = (int(center_x - dx_major), int(center_y - dy_major))
        major_pt2 = (int(center_x + dx_major), int(center_y + dy_major))
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

        valid_drops.append({
            "中心座標 (px)": (int(center_x), int(center_y)),
            "長軸 (cm)": major_axis_cm,
            "短軸 (cm)": minor_axis_cm,
            "旋轉角度 (°)": angle
        })

cv2.imshow('Blood Drop Analysis', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

