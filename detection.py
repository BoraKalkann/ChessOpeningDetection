import cv2
import numpy as np
from ultralytics import YOLO
from openings import openings

# YOLOv8 modelini yükle
model = YOLO("bestSon.pt")

# Tespit edilen taşları saklamak için set ve LİSTE (sıralı)
detected_pieces = set()
detected_pieces_list = []

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
                    if piece not in detected_pieces:
                        detected_pieces.add(piece)
                        detected_pieces_list.append(piece)
                print("Hamle kaydedildi!", move)
                cv2.putText(warped, "HAMLE KAYDEDILDI!", (SIDE//2 - 120, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("♟️ Satranç Tanıma + Grid", warped)
                cv2.waitKey(500)
            else:
                print("Aynı hamle tekrar kaydedilmedi.")
    elif key == 8:  # Backspace tuşu ile hamle geri alma
        if moves:
            last_move = moves.pop()
            print("Son hamle geri alındı!", last_move)
            # detected_pieces_list'ten de son eklenen taşları çıkar
            for piece in last_move:
                if piece in detected_pieces_list:
                    detected_pieces_list.remove(piece)
            # detected_pieces setini de güncelle
            detected_pieces.clear()
            for move in moves:
                for piece in move:
                    detected_pieces.add(piece)
            cv2.putText(warped, "HAMLE GERi ALINDI!", (SIDE//2 - 120, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("♟️ Satranç Tanıma + Grid", warped)
            cv2.waitKey(500)
        else:
            print("Geri alınacak hamle yok.")

# Program sonunda tespit edilen taşları eklenme sırasına göre 2'li notasyon gibi yazdır
print("\nTespit Edilen Taşlar (Notasyon Gibi):")
detected_list = detected_pieces_list  # Sıralı liste
for i in range(0, len(detected_list), 2):
    first = detected_list[i] if i < len(detected_list) else ''
    second = detected_list[i+1] if i+1 < len(detected_list) else ''
    print(f"{i//2+1}. {first}-{second}")

# --- Açılış tespiti ---
# Sadece kare notasyonlarını sırayla al (taş tipiyle birlikte satranç notasyonu üret)
def get_chess_notation(piece_str):
    # Örn: white_bishop_f4 -> Bf4, black_knight_c6 -> Nc6, white_pawn_d4 -> d4
    try:
        color, piece, square = piece_str.split('_')
    except ValueError:
        return piece_str  # Hatalı format
    piece_map = {
        'pawn': '',
        'knight': 'N',
        'bishop': 'B',
        'rook': 'R',
        'queen': 'Q',
        'king': 'K',
    }
    return f"{piece_map.get(piece, '?')}{square}"

opening_moves = []
for i in range(0, len(detected_pieces_list), 2):
    first = detected_pieces_list[i] if i < len(detected_pieces_list) else ''
    second = detected_pieces_list[i+1] if i+1 < len(detected_pieces_list) else ''
    if first:
        move1 = get_chess_notation(first)
        opening_moves.append(move1)
    if second:
        move2 = get_chess_notation(second)
        opening_moves.append(move2)
# Açılış veri tabanı (import openings)
from openings import openings
found = False
for seq, name in openings.items():
    if tuple(opening_moves[:len(seq)]) == seq:
        print(f"Açılış: {name}")
        found = True
        break
if not found:
    print("Açılış tanımlanamadı.")

# Temizlik
cap.release()
cv2.destroyAllWindows()
