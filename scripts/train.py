EPOCHS = 10
MOSAIC = 0.4
OPTIMIZER = 'AdamW'
MOMENTUM = 0.9
LR0 = 0.0001
LRF = 0.0001
SINGLE_CLS = False

import argparse
from ultralytics import YOLO
import os
import sys

if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--mosaic', type=float, default=MOSAIC)
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER)
    parser.add_argument('--momentum', type=float, default=MOMENTUM)
    parser.add_argument('--lr0', type=float, default=LR0)
    parser.add_argument('--lrf', type=float, default=LRF)
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS)
    
    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(this_dir)

    print("\n🚀 Loading YOLO11x Transformer (Attention) model...")
    model = YOLO("yolov9e.pt")     # <-- TRANSFORMER + ATTENTION MODEL

    print("\n📦 Starting Training...\n")

    results = model.train(
        data=os.path.join(os.path.dirname(this_dir), "yolo_params.yaml"),
        epochs=args.epochs,
        device=0, 
        batch=2,                     # GPU
        single_cls=args.single_cls,
        mosaic=args.mosaic,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum
    )

    print("\n🎉 Training Finished!")
    print("➡️ Best model saved at: runs/detect/train/weights/best.pt")
