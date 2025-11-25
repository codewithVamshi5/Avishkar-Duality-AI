# scripts/predict.py
# Colab-ready prediction script for Space_Hackathon_Master
# Reference doc: /mnt/data/Avishkaar Hackathon Documentation.pdf

import os
from pathlib import Path
import yaml
from ultralytics import YOLO
import cv2
import numpy as np

def load_yaml(yaml_path: Path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def select_training_folder(detect_path: Path):
    train_folders = [f for f in sorted(os.listdir(detect_path)) if (detect_path / f).is_dir() and f.startswith("train")]
    if not train_folders:
        raise FileNotFoundError(f"No training folders found in {detect_path}")
    if len(train_folders) == 1:
        return train_folders[0]
    # Multiple training runs found -> ask user to select one
    print("Multiple training runs found. Select the index of the folder to use as model source:")
    for i, f in enumerate(train_folders):
        print(f"  {i}: {f}")
    choice = None
    while choice is None:
        s = input("Enter index (number): ").strip()
        if s.isdigit() and 0 <= int(s) < len(train_folders):
            choice = int(s)
        else:
            print("Invalid choice, try again.")
    return train_folders[choice]

def predict_and_save(model, img_path: Path, out_img_path: Path, out_txt_path: Path, conf=0.5, imgsz=640):
    # Run prediction
    results = model.predict(source=str(img_path), conf=conf, imgsz=imgsz, device=model.device)
    res = results[0]

    # Plot + save image with boxes
    try:
        plot_img = res.plot()  # returns numpy image
        # Ensure BGR for cv2.imwrite if needed; ultralytics usually returns rgb. Convert to BGR for cv2
        if isinstance(plot_img, np.ndarray):
            bgr = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
        else:
            bgr = plot_img
        cv2.imwrite(str(out_img_path), bgr)
    except Exception as e:
        print(f"Warning: could not save plotted image for {img_path}: {e}")

    # Save normalized bounding boxes in YOLO format: class x_center y_center width height
    # Use result.boxes.xywhn (normalized) and result.boxes.cls
    boxes_xywhn = None
    classes = None
    try:
        boxes_xywhn = res.boxes.xywhn.cpu().numpy() if hasattr(res.boxes, "xywhn") else None
        classes = res.boxes.cls.cpu().numpy() if hasattr(res.boxes, "cls") else None
    except Exception:
        # fallback: try to extract per-box
        try:
            boxes_xywhn = np.array([b.xywhn[0].cpu().numpy() for b in res.boxes])
            classes = np.array([int(b.cls.item()) for b in res.boxes])
        except Exception:
            boxes_xywhn = None
            classes = None

    with open(out_txt_path, "w") as f:
        if boxes_xywhn is None or classes is None:
            # nothing detected
            pass
        else:
            for c, xy in zip(classes, boxes_xywhn):
                # xy is normalized [x_center, y_center, width, height]
                x_center, y_center, width, height = map(float, xy)
                f.write(f"{int(c)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def main():
    this_dir = Path(__file__).resolve().parent
    # yaml is one folder above scripts/
    yaml_path = this_dir.parent / "yolo_params.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"yolo_params.yaml not found at expected location: {yaml_path}")

    cfg = load_yaml(yaml_path)

    if 'test' in cfg and cfg['test']:
        test_images_dir = Path(cfg['test']) / "images" if (Path(cfg['test']) / "images").exists() else Path(cfg['test'])
    else:
        raise ValueError("No 'test' path found in yolo_params.yaml; please add it pointing to the test images folder")

    if not test_images_dir.exists():
        raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")

    # Find the model weights inside runs/detect/train*/
    detect_path = this_dir / "runs" / "detect"
    if not detect_path.exists():
        raise FileNotFoundError(f"No runs/detect directory found at {detect_path}. Have you trained yet?")

    train_folder = select_training_folder(detect_path)
    model_path = detect_path / train_folder / "weights" / "best.pt"
    if not model_path.exists():
        # fallback to last.pt
        model_path = detect_path / train_folder / "weights" / "last.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"No model weights found in {detect_path / train_folder / 'weights/'}")

    print(f"Using model: {model_path}")
    model = YOLO(str(model_path))  # will pick GPU automatically if available

    # Prepare output folders
    out_root = this_dir / "predictions"
    images_out = out_root / "images"
    labels_out = out_root / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # iterate images
    allowed_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([p for p in test_images_dir.iterdir() if p.suffix.lower() in allowed_ext])
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in test directory {test_images_dir}")

    print(f"Found {len(image_files)} test images. Running predictions...")

    for img_path in image_files:
        out_img = images_out / img_path.name
        out_txt = labels_out / img_path.with_suffix('.txt').name
        predict_and_save(model, img_path, out_img, out_txt, conf=0.5, imgsz=640)

    print(f"\n✅ Predictions saved:\n Images -> {images_out}\n Labels -> {labels_out}")

    # Finally run evaluation on the test split (produces mAP, confusion matrix, PR curves)
    print("\n📊 Running final evaluation (model.val) on test split...")
    try:
        val_results = model.val(data=str(yaml_path), split="test", imgsz=640, device=model.device, plots=True)
        print("\n✅ Evaluation complete. Metrics summary:")
        print(val_results.metrics if hasattr(val_results, "metrics") else val_results)
    except Exception as e:
        print(f"Warning: model.val() failed: {e}")

if __name__ == "__main__":
    main()
