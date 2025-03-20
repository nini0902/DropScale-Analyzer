import cv2
import numpy as np

def detect_red_hsv_range(image_path):
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or could not be loaded.")
        return
    
    # 轉換成 HSV 色彩空間
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定義紅色範圍（兩段）
    lower_red1 = np.array([0, 50, 50])  # 低飽和度範圍
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])  # 高飽和度範圍
    upper_red2 = np.array([180, 255, 255])

    # 建立遮罩，過濾紅色區域
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # 找出紅色區域的 HSV 值
    red_pixels = hsv[mask > 0]  # 只取紅色區域的 HSV 值

    if len(red_pixels) == 0:
        print("No red areas detected.")
        return

    # 計算 HSV 的最小與最大範圍
    min_hsv = np.min(red_pixels, axis=0)
    max_hsv = np.max(red_pixels, axis=0)

    print(f"紅色區域的 HSV 範圍：")
    print(f"最小值：H={min_hsv[0]}, S={min_hsv[1]}, V={min_hsv[2]}")
    print(f"最大值：H={max_hsv[0]}, S={max_hsv[1]}, V={max_hsv[2]}")

# 使用範例（請修改成你的圖片路徑）
image_path = r"C:\Users\st\Downloads\60.png"
detect_red_hsv_range(image_path)

