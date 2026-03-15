from pathlib import Path
import random
import cv2

ROOT = Path("data/processed/cowc_yolo_patch")
SPLIT = "train"


def main():
    image_dir = ROOT / "images" / SPLIT
    label_dir = ROOT / "labels" / SPLIT

    images = list(image_dir.glob("*"))
    if not images:
        print("No images found.")
        return

    sample_images = random.sample(images, min(12, len(images)))

    for img_path in sample_images:
        label_path = label_dir / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read image: {img_path}")
            continue

        h, w = img.shape[:2]
        print(f"\nImage: {img_path.name}")
        print(f"Label: {label_path.name}")

        if label_path.exists():
            lines = label_path.read_text(encoding="utf-8").splitlines()
            print(f"Lines in label file: {len(lines)}")

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                cls, xc, yc, bw, bh = map(float, parts)
                x1 = int((xc - bw / 2) * w)
                y1 = int((yc - bh / 2) * h)
                x2 = int((xc + bw / 2) * w)
                y2 = int((yc + bh / 2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(
                    img,
                    f"class {int(cls)}",
                    (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        else:
            print("Label file missing.")

        cv2.imshow(img_path.name, img)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == 27:  # ESC to stop
            break


if __name__ == "__main__":
    main()