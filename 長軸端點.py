import cv2
import numpy as np

# 讀取影像
image = cv2.imread(r"C:\Users\user\Downloads\60.png")

# 轉換為灰階
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# **顯示灰階影像**
cv2.imshow('Gray Image', gray)
cv2.waitKey(0)

# **使用二值化處理，區分血滴與背景**
_, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

# **去除小雜訊**
kernel = np.ones((5, 5), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# **找血滴輪廓**
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    if len(cnt) >= 5:
        # **擬合橢圓**
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon, True)
        ellipse = cv2.fitEllipse(cnt)
        (center_x, center_y), (major_axis_px, minor_axis_px), angle = ellipse

        # **確保 major_axis 為長軸**
        if minor_axis_px > major_axis_px:
            major_axis_px, minor_axis_px = minor_axis_px, major_axis_px
            angle += 90  # 旋轉 90° 確保長軸方向正確

        # **計算長軸端點**
        angle_rad = np.deg2rad(angle)
        dx_major = (major_axis_px / 2) * np.cos(angle_rad)
        dy_major = (major_axis_px / 2) * np.sin(angle_rad)

        major_pt1 = (int(center_x - dx_major), int(center_y - dy_major))
        major_pt2 = (int(center_x + dx_major), int(center_y + dy_major))

        # **標記長軸的兩個點 (紅色)**
        cv2.circle(image, major_pt1, 5, (0, 0, 255), -1)  # 紅色點
        cv2.circle(image, major_pt2, 5, (0, 0, 255), -1)  # 紅色點

# **顯示最終影像**
cv2.imshow('Detected Major Axis Points (Gray Mode)', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

