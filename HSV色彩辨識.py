import cv2
import numpy as np

def show_hsv_on_click(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or could not be loaded.")
        return

    # 縮小顯示圖片
    scale_percent = 50
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height))
    hsv_resized = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

    display_image = resized_image.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            hsv_val = hsv_resized[y, x]
            h, s, v = hsv_val
            print(f"Clicked at ({x}, {y}) → H: {h}, S: {s}, V: {v}")

            # 畫圓 & 顯示 HSV 文字
            cv2.circle(display_image, (x, y), 5, (255, 0, 0), -1)
            text = f"H:{h} S:{s} V:{v}"
            cv2.putText(display_image, text, (x + 10, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Click to Get HSV", display_image)

    cv2.namedWindow("Click to Get HSV", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Click to Get HSV", width, height)
    cv2.setMouseCallback("Click to Get HSV", mouse_callback)
    cv2.imshow("Click to Get HSV", display_image)

    while True:
        key = cv2.waitKey(1)
        if key == 27:  # 按 ESC 離開
            break

    cv2.destroyAllWindows()

# 使用範例
image_path = r"C:\Users\nicole\Downloads\60.png"
show_hsv_on_click(image_path)

