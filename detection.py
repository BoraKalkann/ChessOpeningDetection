import cv2
from ultralytics import YOLO
import time

# YOLOv8 modelini yükle
model = YOLO("bestSon.pt")  # GoogleCollab'de eğitilen en iyi ağırlıklar

# DroidCam ile gömülü webcam'den kurturulyoruz.
cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Kameraya erişilemiyor.")
    exit()

print("🎥 Tanıma başlatıldı.'q' tuşuna basarak çıkabilirsiniz.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Görüntü alınamadı.")
        break

    # YOLOv8 eğitilen model ile tahmin yapılması
    results = model(frame)

    # Tespit edilen taşları dikdörtgen içine alarak label sağla.
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        for i in range(len(coords)):
            x1, y1, x2, y2 = map(int, coords[i])
            confidence = float(confs[i])
            class_id = int(classes[i])
            class_name = model.names[class_id]
            label = f"{class_name} {confidence:.2f}"

            # Kutuç çiz ve etiketle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow("♟️ Satranç Taşı Tanıma", frame)

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()
