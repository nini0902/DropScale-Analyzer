import cv2
import numpy as np

def detect_black_line_length(image):
    """ 辨識黑色粗線的長度，並作為比例尺 (1 公分) """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)  # 反轉黑白，找黑色區域

    # 找輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_length = 0
    best_rect = None

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > max_length and h < 20:  # 假設黑色線條比較長，但不太寬
            max_length = w
            best_rect = (x, y, w, h)

    if best_rect:
        x, y, w, h = best_rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)  # 藍色框標記黑線
        cv2.putText(image, "1 cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
        return w  # 回傳黑線的長度（像素）
    return None

def detect_red_regions(image_path):
    # 讀取圖片
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or could not be loaded.")
        return

    # 取得黑色比例尺長度
    black_line_length = detect_black_line_length(image)

    if black_line_length is None:
        print("Error: No black scale line detected.")
        return

    # 轉換成 HSV 色彩空間
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 設定紅色範圍
    lower_red1 = np.array([0, 50, 154])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 50, 154])
    upper_red2 = np.array([180, 255, 255])

    # 建立遮罩，合併兩段紅色範圍
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    # 形態學運算去除雜訊
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 找輪廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原圖上標示紅色圓點
    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (x, y), (major_axis, minor_axis), angle = ellipse  # 取得橢圓參數

            # 計算紅點的長寬，並轉換成 cm
            length_cm = (major_axis / black_line_length)  # 長軸換算為 cm
            width_cm = (minor_axis / black_line_length)   # 短軸換算為 cm

            # 忽略小於 1.1 cm 的紅點
            if length_cm < 1.1 or width_cm < 1.1:
                continue

            # 標示紅點的大小（藍色字體）
            text = f"{length_cm:.2f} cm x {width_cm:.2f} cm"
            cv2.putText(image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 
                        2, (255, 0, 0), 6)

            # 繪製橢圓
            cv2.ellipse(image, ellipse, (255, 0, 0), 4)

    # 顯示圖片
    cv2.imshow("Detected Red Regions", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用範例（請修改成你的圖片路徑）
image_path = r"C:\Users\st\Downloads\60.png"
detect_red_regions(image_path)

