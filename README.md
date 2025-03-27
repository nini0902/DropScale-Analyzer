## 🛠 需求環境
請確保你的環境已安裝以下 **Python 套件**：
```bash
pip install opencv-python numpy
```
```bash
install opencv-python
```
```bash
install mediapipe
```
## 🖼 圖片需要長怎樣
1. 用 Canva 之類的，在畫面中標示一個黑色線段當成一公分的比例尺 ( 我是用畫面裡的尺去畫的 ) 
2. 這個事建議，但畫面盡可能的越乾淨越好哦 ouob

![圖片預處理](result_demo.png)
## 🚀 使用方法
 ( 建議在 thonny 或是 vscode 執行 ) 
 ### 1️⃣ 輸入影像
請將影像存放至你的電腦，並修改 image_path 變數：
```bash
image_path = r"(改成你的圖片的路徑)"
```
 ### 2️⃣ 執行程式
請執行以下指令：
 ```bash
python detect.py
```
### 3️⃣ 查看結果
結果將會顯示在視窗：
- Result：標記紅點與藍線的影像
- Red Mask：紅色遮罩影像
程式會將結果儲存到 桌面 (Desktop)
### 4️⃣ 輸出數據
執行後，終端機會輸出紅點的實際尺寸，例如：
```bash
圓點1：1.70 cm x 1.83 cm
圓點2：1.86 cm x 1.95 cm
圓點3：1.83 cm x 2.04 cm
```
## 📸 結果展示
結果就會標示長寬公分數 ( 辨識結果小於0.9公分會視為其他噴濺雜物，會自動忽略哦，不想忽略的直接改 code 就好 ) 
![結果展示](25(1).png)
