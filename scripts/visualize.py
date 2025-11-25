import os
import cv2
from pathlib import Path

class YoloVisualizer:
    MODE_TRAIN = "train"
    MODE_VAL = "val"
    MODE_TEST = "test"

    def __init__(self, project_root):
        self.project_root = Path(project_root)

        # Path to classes.txt
        classes_file = self.project_root / "classes.txt"
        if not classes_file.exists():
            raise FileNotFoundError("classes.txt not found at project root.")

        self.classes = classes_file.read_text().splitlines()
        self.class_map = {i: c for i, c in enumerate(self.classes)}

        self.set_mode(self.MODE_TRAIN)

        # Output folder to save visualization results
        self.output_dir = self.project_root / "visualized"
        self.output_dir.mkdir(exist_ok=True)

    def set_mode(self, mode):
        dataset_dir = self.project_root / "dataset" / mode

        self.images_folder = dataset_dir / "images"
        self.labels_folder = dataset_dir / "labels"

        self.image_names = sorted([f for f in os.listdir(self.images_folder) if f.endswith((".jpg", ".png"))])
        self.label_names = sorted(os.listdir(self.labels_folder))

        assert len(self.image_names) == len(self.label_names), "Mismatch between images and labels!"
        assert len(self.image_names) > 0, "No images found."

        self.num_images = len(self.image_names)
        self.frame_index = 0

        print(f"📁 Mode set to {mode.upper()} — {self.num_images} images found.")

    def visualize_frame(self, idx):
        img_path = self.images_folder / self.image_names[idx]
        label_path = self.labels_folder / self.label_names[idx]

        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]

        # Read bounding boxes
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    cls, x, y, bw, bh = map(float, line.split())
                    cls = int(cls)

                    # Convert normalized YOLO → pixel coordinates
                    cx = int(x * w)
                    cy = int(y * h)
                    bw = int(bw * w)
                    bh = int(bh * h)

                    x1 = cx - bw // 2
                    y1 = cy - bh // 2
                    x2 = x1 + bw
                    y2 = y1 + bh

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, self.class_map[cls], (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Save output
        save_path = self.output_dir / f"vis_{self.image_names[idx]}"
        cv2.imwrite(str(save_path), image)

        print(f"✔ Saved: {save_path}")
        return image

    def visualize_all(self):
        print(f"🔍 Visualizing all images...")
        for i in range(self.num_images):
            self.visualize_frame(i)

        print(f"\n🎉 Visualization complete!")
        print(f"All visualized images saved to: {self.output_dir}")


if __name__ == "__main__":
    # Auto detect project root (one level above scripts/)
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    vis = YoloVisualizer(project_root)

    # Change modes as needed:
    # vis.set_mode(YoloVisualizer.MODE_TRAIN)
    # vis.set_mode(YoloVisualizer.MODE_VAL)
    # vis.set_mode(YoloVisualizer.MODE_TEST)

    vis.visualize_all()
