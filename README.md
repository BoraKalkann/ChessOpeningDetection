   ♟️ Real-Time Chess Move Tracker
   
   📖 About the Project
   This project uses YOLOv8 to detect chess pieces in real-time and track moves on a physical chessboard. The system records moves and matches them against known openings to display the identified opening.
   
   Goals:
    Analyze piece positions
    Automatically record moves
    Detect and display chess openings visually
   
   ⚡ Features:
   - 🎥 Real-time piece detection
   - 📝 Record moves in standard chess notation (e.g., Nf3, e4)
   - ⏮️ Undo and replay moves
   - 📚 Recognize known chess openings
   - 🖼 Manual board corner selection and grid visualization
   - 💡 User-friendly visual feedback
   
   🛠 Requirements:
   -ultralytics 
   -torch 
   -torchvision 
   -opencv-python
   -numpy 
   -matplotlib 
   -pyyaml
   -droidcam



  🧩 How It Works
  The camera captures the board and applies a perspective transform.
  YOLOv8 model detects the pieces.
  Piece centers are mapped to chessboard squares.
  Moves are recorded, ignoring duplicates.
  Recorded moves are compared against a known openings database.
  Detected pieces and openings are displayed in OpenCV windows.
  
  🖼 Visualization
  Blue grid for board squares
  Green rectangles highlight detected pieces with confidence scores
  Square names displayed in blue text
  Detected moves and openings displayed in a popup window

 How To Make It:
  First you have to download DroidCam.(You can download the application from playstore)
  Then you have to do some settings work to use your phone as a webcam(You can find the information on the internet.)
  Once you completed that make sure you have the correct webcam (cap = cv2.VideoCapture(2) it can be 0 1 or 2. For me it's 2.)
  The important part is what chess board you are going to use and which dataset your model is going to be trained on.
  Lucky for me the chess board that I had is the same chess board in the dataset :DDDD
  So the confidence and correctness is going to be higher.
  The dataset I used is:
  Chess Piece detection v2 Computer Vision Project - Pannon Egyetem
  Great dataset with lots of labels and positions.

  The dataset contains 832 chess images labeled with chess.
  Horizontal and Vertical Flip - Stretch Operations
  However, the total number of data obtained after the augmentation process is
  1500. (1168 Train Sets (78%), 167 Valid Sets (11%), 165 Test Sets (11%))
  The augmentation process was performed in Roboflow.
  There are 12 Label Classes:
  <img width="1114" height="72" alt="image" src="https://github.com/user-attachments/assets/385cfd1c-9c84-46e7-b4d5-a66ab5e7581c" />

  I trained the model in GoogleCollab for the soul reason to be more efficient.(and my PC is garbage)
  This is the code for the training process:
  
  !pip install ultralytics
  import zipfile
  With
  zipfile.ZipFile('/content/Chess Piece detection v2.v1i.yolov8.zip', 'r') as zip_ref:
  zip_ref.extractall('/content/hedef_klasor')
  from ultralytics import YOLO
  model = YOLO('yolov8n.pt')
  results = model.train(
   data='data.yaml',
   epochs=100,
   imgsz=640,
   batch=8, 
   patience=7,
   device=0,
   augment=True,
   save_period=5,
   project='/content/drive/MyDrive/yolov8_runs',
   name='chess_exp_aug',
   exist_ok=True

   WARNİNG! DO NOT FORGET TO UPDATE THE DATA.YAML 
   <img width="1320" height="325" alt="image" src="https://github.com/user-attachments/assets/ee0abbd1-492a-484f-86ef-2ce89052dc0c" />
   In this section, it's crucial to provide the paths to the train, val, and test image files, otherwise learning won't begin.
   After training, a best.pt extension containing the best weights is generated.
   Other weights are also available, such as last.pt (last epoch weights).

   Here are some graphs for my nerd friends out there:
   <img width="1386" height="755" alt="image" src="https://github.com/user-attachments/assets/f3f67c9d-f8b4-4744-b811-68dd0a6d3a05" />
   <img width="1404" height="937" alt="image" src="https://github.com/user-attachments/assets/e4f0c821-9f02-471a-a308-62c2fb4d2ed1" />
   <img width="1480" height="767" alt="image" src="https://github.com/user-attachments/assets/c30a86f8-89ea-4f9b-a0ae-17f6da4aa0b2" />

   If you didn't get a good confidence score thats because of the dataset that you used. I do not have a another dataset for you to train but thats your job now.

   For the project results you can checkout my youtube channel folks.
   https://www.youtube.com/channel/UCZLl5d5DlNKBwl7dT_RkFyA

   Thank you for your interest and God bless.
   If you couldn't find a good dataset God may have mercy on your soul...

















