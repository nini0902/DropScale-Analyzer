import cv2
import numpy as np

# === 讀取轉存後的圖片 ===
file_path = r"C:\test\25.jpg"  # 確保這是轉存後的標準 RGB 圖片
img = cv2.imread(file_path)

# 確保圖片成功讀取
if img is None:
    print("❌ 讀取影像失敗，請確認影像檔案路徑。")
    exit()

# 建立影像拷貝（避免修改原圖）
img_annotated = img.copy()

# === 轉換為 HSV 色域，偵測紅色 ===
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 紅色範圍（兩段）
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

# 建立紅色遮罩
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = mask1 + mask2  # 合併兩個紅色範圍

# === 找出紅色區域的輪廓 ===
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 讓使用者輸入比例尺長度（像素對應的實際公分）
scale_px = float(input("請輸入比例尺在照片中的像素長度（例如 50 像素對應 1 cm）: "))
scale_cm = 1.0  # 假設比例尺代表 1 cm
pixel_to_cm = scale_cm / scale_px  # 每個像素對應的公分數

print("====== 偵測到的紅色圓點 ======")
idx = 1

for cnt in contours_red:
    # 忽略過小噪點
    area = cv2.contourArea(cnt)
    if area < 10:
        continue

    # 計算最小外接圓
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    circle_area = np.pi * (radius ** 2)
    
    # 檢查該區域是否接近圓形（避免雜訊或非圓形物件）
    if abs(circle_area - area) / circle_area > 0.3:  # 誤差大於 30% 就忽略
        continue

    # 轉換為整數
    x, y, radius = int(x), int(y), int(radius)

    # 計算圓的直徑並轉換為公分，取小數點後一位
    diameter_px = 2 * radius
    diameter_cm = round(diameter_px * pixel_to_cm, 1)

    print(f"紅色圓點 {idx}: 直徑 = {diameter_cm} cm")

    # 繪製圓形外框（深綠色）
    cv2.circle(img_annotated, (x, y), radius, (0, 128, 0), 3)

    # 標示直徑長度
    text = f"#{idx} {diameter_cm} cm"
    cv2.putText(img_annotated, text, (x - radius, y - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)

    # 標示長寬線條
    cv2.line(img_annotated, (x - radius, y), (x + radius, y), (255, 140, 0), 3)  # 橘色
    cv2.line(img_annotated, (x, y - radius), (x, y + radius), (0, 0, 255), 3)  # 藍色

    idx += 1

# === 顯示與存檔結果 ===
cv2.namedWindow("Annotated Image", cv2.WINDOW_NORMAL)  # 設定視窗可調整大小
cv2.resizeWindow("Annotated Image", 800, 600)         # 固定初始大小
cv2.imshow("Annotated Image", img_annotated)          # 顯示標示後的影像
cv2.imwrite("annotated_result.jpg", img_annotated)    # 存檔標示後的影像
cv2.waitKey(0)
cv2.destroyAllWindows()

