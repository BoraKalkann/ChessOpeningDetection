import cv2
import numpy as np
from ultralytics import YOLO
from openings import openings

model = YOLO("bestSon.pt")

def compute_iou(box, boxes):
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    union = box_area + boxes_area - intersection
    iou = intersection / union
    return iou
def get_chess_notation(piece_str):
   
    try:
        color, piece, square = piece_str.split('_')
    except ValueError:
        return piece_str  
    piece_map = {
        'pawn': '',
        'knight': 'N',
        'bishop': 'B',
        'rook': 'R',
        'queen': 'Q',
        'king': 'K',
    }
    return f"{piece_map.get(piece, '?')}{square}"

detected_pieces = set()
detected_pieces_list = []
moves = []
def get_square_position(x, y, cell_size):
    col = x // cell_size
    row = y // cell_size
    if 2 <= row <= 5 and 0 <= col <= 7:  
        return f"{chr(97 + int(col))}{8 - int(row)}"
    return None

cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print("Kameraya erişilemiyor.")
    exit()

print("📌 Tahtanın 4 köşesini sırayla tıklayın: [Sol Üst, Sağ Üst, Sağ Alt, Sol Alt]")

points = []

def select_corner(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))


frame_ready = False
while not frame_ready:
    ret, frame = cap.read()
    if ret and frame is not None:
        frame_ready = True
    else:
        print("📷 Kamera görüntüsü bekleniyor...")


window_name = "Kose Secimi"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, select_corner)


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    clone = frame.copy()
    for pt in points:
        cv2.circle(clone, pt, 5, (0, 0, 255), -1)

    cv2.imshow(window_name, clone)
    if cv2.waitKey(1) & 0xFF == 27 or len(points) == 4: 
        break

cv2.destroyWindow(window_name)

# 2️⃣ PERSPEKTİF DÖNÜŞÜMÜ
SIDE = 480  
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
    results = model(warped)
    current_pieces = set()
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        
        # Non-maximum suppression uygula
        selected_indices = []
        for cls in np.unique(classes):
            # Her sınıf için ayrı NMS
            cls_mask = classes == cls
            cls_boxes = coords[cls_mask]
            cls_scores = confs[cls_mask]
            
            
            keep = []
            indices = np.argsort(cls_scores)[::-1]
            while len(indices) > 0:
                idx = indices[0]
                keep.append(idx)
                if len(indices) == 1:
                    break
                    
                
                box1 = cls_boxes[idx]
                other_boxes = cls_boxes[indices[1:]]
                ious = compute_iou(box1, other_boxes)
                
                
                indices = indices[1:][ious < 0.3]
                
            selected_indices.extend(np.where(cls_mask)[0][keep])
            
       
        coords = coords[selected_indices]
        confs = confs[selected_indices]
        classes = classes[selected_indices]

        for i in range(len(coords)):
            x1, y1, x2, y2 = map(int, coords[i])
            confidence = float(confs[i])
            class_id = int(classes[i])
            class_name = model.names[class_id]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            square = get_square_position(center_x, center_y, SIDE // 8)
            if square and confidence > 0.8:
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
      
        move = sorted(current_pieces)
        if move:
           
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
    elif key == ord('a'): 
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
        found = False
        opening_name = "Açılış tanımlanamadı"
        max_match_length = 0
        
        for seq, name in openings.items():
            if len(seq) <= len(opening_moves) and tuple(opening_moves[:len(seq)]) == seq:
                if len(seq) >= max_match_length:
                    max_match_length = len(seq)
                    opening_name = name
                    found = True
        
        popup = np.zeros((400, 600, 3), dtype=np.uint8)
       
        cv2.putText(popup, "Tespit Edilen Acilis:", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
        cv2.putText(popup, opening_name, (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                    
        # Tespit edilen taşları ekle
        cv2.putText(popup, "Tespit Edilen Taslar:", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
        y_offset = 200
        for i in range(0, len(detected_pieces_list), 2):
            first = detected_pieces_list[i] if i < len(detected_pieces_list) else ''
            second = detected_pieces_list[i+1] if i+1 < len(detected_pieces_list) else ''
            move_text = f"{i//2+1}. {first}-{second}"
            cv2.putText(popup, move_text, (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            y_offset += 30
            
        popup_window = "Tespit Edilen Açılış ve Taşlar"
        cv2.imshow(popup_window, popup)
        cv2.waitKey(5000)  # 5 saniye göster
        try:
            cv2.destroyWindow(popup_window)
        except cv2.error:
            pass
    elif key == 8:  
        if moves:
            last_move = moves.pop()
            print("Son hamle geri alındı!", last_move)
            
            for piece in last_move:
                if piece in detected_pieces_list:
                    detected_pieces_list.remove(piece)
           
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


print("\nTespit Edilen Taşlar :")
detected_list = detected_pieces_list  # Sıralı liste
for i in range(0, len(detected_list), 2):
    first = detected_list[i] if i < len(detected_list) else ''
    second = detected_list[i+1] if i+1 < len(detected_list) else ''
    print(f"{i//2+1}. {first}-{second}")

# --- Açılış tespiti ---

def get_chess_notation(piece_str):
   
    try:
        color, piece, square = piece_str.split('_')
    except ValueError:
        return piece_str  
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

from openings import openings
found = False
for seq, name in openings.items():
    if tuple(opening_moves[:len(seq)]) == seq:
        print(f"Açılış: {name}")
        found = True
        break
if not found:
    print("Açılış tanımlanamadı.")


cap.release()
cv2.destroyAllWindows()
