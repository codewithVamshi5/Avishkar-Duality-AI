Space_Hackathon_Master/
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── classes.txt
├── yolo_params.yaml
│
├── scripts/
│   ├── train.py
│   ├── predict.py
│   ├── visualize.py
│   │
│   └── runs/                  # training outputs are saved here
│        └── detect/
│        |    └── train3/
│        |       └── weights/
│        |           ├── best.pt                    //.the mode we have trained
│        |           └── last.pt
|        |           |__ app.py
         |
         |__ metrics.png's
