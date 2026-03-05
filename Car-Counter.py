from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture("data/cars.mp4")  

model = YOLO("yolov8n.pt") 

# 定義 COCO 資料集的 80 個類別名稱
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    # 4. 進行影像推理，使用 stream=True 提高處理效率
    if not success:
        print("影片讀取失敗或已播放結束。")
        break

    results = model(img, stream=True)

    # 5. 尋訪結果並提取邊界框、信心度與類別
    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            # 獲取邊界框座標 X1, Y1, X2, Y2
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 計算寬度 (w) 與高度 (h) 給 cvzone 使用
            w, h = x2 - x1, y2 - y1
            
            # 繪製角落風格的矩形邊界框
            cvzone.cornerRect(img, (x1, y1, w, h), l=15)
            
            # 計算信心度 (四捨五入到小數點後兩位)
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # 取得預測類別 ID，並轉換為類別名稱
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            
            # 在邊界框上方顯示類別與信心度，設定 max 防止文字超出視窗邊緣
            cvzone.putTextRect(img, f'{currentClass} {conf}', 
                               (max(0, x1), max(35, y1)), 
                               scale=0.6, thickness=1, offset=3)

cv2.imshow("Image", img)
cv2.waitKey(1)

    