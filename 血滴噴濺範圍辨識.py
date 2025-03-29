import cv2
import numpy as np
import os
import ctypes.wintypes

def get_desktop_path():
    CSIDL_DESKTOPDIRECTORY = 0x10
    SHGFP_TYPE_CURRENT = 0
    buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
    ctypes.windll.shell32.SHGetFolderPathW(None, CSIDL_DESKTOPDIRECTORY, None, SHGFP_TYPE_CURRENT, buf)
    return buf.value

def detect_black_scale(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 50, 80]))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = max((cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= 100),
               key=lambda r: max(r[2], r[3]), default=None)
    return max(best[2], best[3]) if best else None

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

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
    mask1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 80, 50]), np.array([180, 255, 255]))
    mask = mask1 + mask2

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(cnt)
        regions.append({
            "contour": cnt,
            "center": (cx, cy),
            "w_cm": w / black_line_length,
            "h_cm": h / black_line_length,
            "used": False
        })

    big = [r for r in regions if r["w_cm"] >= 0.7 or r["h_cm"] >= 0.7]
    small = [r for r in regions if r["w_cm"] < 0.7 and r["h_cm"] < 0.7]

    groups = []
    for r1 in big:
        if r1["used"]:
            continue
        group = [r1]
        r1["used"] = True
        for r2 in big:
            if not r2["used"] and distance(r1["center"], r2["center"]) < 100:
                group.append(r2)
                r2["used"] = True
        groups.append(group)

    for sr in small:
        nearest = min(groups, key=lambda g: min(distance(sr["center"], r["center"]) for r in g))
        nearest.append(sr)

    drop_count = 0
    for group in groups:
        drop_count += 1
        all_pts = np.concatenate([r["contour"] for r in group])
        x, y, w, h = cv2.boundingRect(all_pts)
        w_cm = w / black_line_length
        h_cm = h / black_line_length
        text = f"Drop {drop_count}: {w_cm:.2f} cm x {h_cm:.2f} cm"
        print(text)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    extra_space = 100
    extended = np.full((image.shape[0] + extra_space, image.shape[1], 3), 255, dtype=np.uint8)
    extended[:image.shape[0], :] = image

    desktop = get_desktop_path()
    output_path = os.path.join(desktop, "output_result.png")
    cv2.imwrite(output_path, extended)
    print("圖片已儲存到：", output_path)

    scale = 0.4
    resized = cv2.resize(extended, (int(extended.shape[1] * scale), int(extended.shape[0] * scale)))
    cv2.imshow("Detected Blood Drops", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 圖片路徑請自行修改
image_path = r"C:\Users\p0966\Downloads\90.png"
detect_red_regions_and_black(image_path)

