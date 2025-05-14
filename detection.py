import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 modelini yükle
model = YOLO("bestSon.pt")

# Tespit edilen taşları saklamak için set 
detected_pieces = set()

# Hamleleri sırayla tutmak için liste
moves = []

# Bir taşın hangi karede olduğunu bulmak için yardımcı fonksiyon
def get_square_position(x, y, cell_size):
    col = x // cell_size
    row = y // cell_size
    if 2 <= row <= 5 and 0 <= col <= 7:  # Sadece 3-6 satırlar 
        return f"{chr(97 + int(col))}{8 - int(row)}"
    return None

# Kamera bağlantısı 
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Kameraya erişilemiyor.")
    exit()

print("📌 Tahtanın 4 köşesini sırayla tıklayın: [Sol Üst, Sağ Üst, Sağ Alt, Sol Alt]")

# 1️⃣ KÖŞE SEÇİMİ
points = []

def select_corner(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))

# Kamera düzgün kare verene kadar bekle
frame_ready = False
while not frame_ready:
    ret, frame = cap.read()
    if ret and frame is not None:
        frame_ready = True
    else:
        print("📷 Kamera görüntüsü bekleniyor...")

# Pencere oluştur ve callback tanımla
window_name = "Kose Secimi"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, select_corner)

# Seçim döngüsü
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    clone = frame.copy()
    for pt in points:
        cv2.circle(clone, pt, 5, (0, 0, 255), -1)

    cv2.imshow(window_name, clone)
    if cv2.waitKey(1) & 0xFF == 27 or len(points) == 4:  # ESC tuşu veya 4 köşe tamam
        break

cv2.destroyWindow(window_name)

# 2️⃣ PERSPEKTİF DÖNÜŞÜMÜ
SIDE = 480  # Düzleştirilen kare boyutu
src = np.array(points, dtype="float32")
dst = np.array([[0, 0], [SIDE, 0], [SIDE, SIDE], [0, SIDE]], dtype="float32")
matrix = cv2.getPerspectiveTransform(src, dst)

# 3️⃣ ALGILAMA VE GRID ÇİZİMİ
print("🎥 Tanıma başladı. 'q' tuşu ile çıkabilirsiniz.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    warped = cv2.warpPerspective(frame, matrix, (SIDE, SIDE))

    # YOLOv8 ile tahmin
    results = model(warped)
    current_pieces = set()
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
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            square = get_square_position(center_x, center_y, SIDE // 8)
            if square and confidence > 0.7:
                piece_info = f"{class_name}_{square}"
                current_pieces.add(piece_info)
            if 2 * (SIDE // 8) <= center_y <= 6 * (SIDE // 8):
                label = f"{class_name} {confidence:.2f}"
                cv2.rectangle(warped, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(warped, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if square:
                    cv2.putText(warped, square, (center_x - 10, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Grid çizimi ve kare isimleri (sadece 3-6 satırlar için)
    cell_size = SIDE // 8
    for row in range(2, 6):
        for col in range(8):
            x1 = col * cell_size
            y1 = row * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            cv2.rectangle(warped, (x1, y1), (x2, y2), (255, 0, 0), 1)
            square_name = f"{chr(97 + col)}{8 - row}"
            cv2.putText(warped, square_name, (x1 + 2, y1 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Tespit edilen taşları ekranın üstünde göster
    y_offset = 30
    cv2.putText(warped, "Tespit Edilen Taslar:", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for idx, piece in enumerate(sorted(detected_pieces)):
        y_pos = y_offset + (idx + 1) * 20
        cv2.putText(warped, piece, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("♟️ Satranç Tanıma + Grid", warped)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        # Sadece space'e basıldığında o anki taşları ekle
        move = sorted(current_pieces)
        if move:
            # Son kaydedilen hamle ile aynıysa ekleme
            if not moves or move != moves[-1]:
                moves.append(move)
                for piece in move:
                    detected_pieces.add(piece)
                print("Hamle kaydedildi!", move)
                cv2.putText(warped, "HAMLE KAYDEDILDI!", (SIDE//2 - 120, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("♟️ Satranç Tanıma + Grid", warped)
                cv2.waitKey(500)
            else:
                print("Aynı hamle tekrar kaydedilmedi.")

# Program sonunda hamleleri satranç maçlarındaki gibi yazdır
print("\nHamleler:")
# Her hamlede sadece değişen taşları ve kareleri göster
for i in range(0, len(moves), 2):
    white = ''
    black = ''
    if i < len(moves):
        # Sadece ilk hamledeki taşları göster
        white_moves = [piece.split('_')[1] for piece in moves[i]]
        white = ' '.join(white_moves)
    if i+1 < len(moves):
        black_moves = [piece.split('_')[1] for piece in moves[i+1]]
        black = ' '.join(black_moves)
    print(f"{i//2+1}. {white} {black}")

# Program sonunda tespit edilen taşları yazdır
print("\nTespit Edilen Taşlar:")
for piece in sorted(detected_pieces):
    print(piece)

# Temizlik
cap.release()
cv2.destroyAllWindows()
