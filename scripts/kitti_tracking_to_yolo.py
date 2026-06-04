#!/usr/bin/env python3
import os
from pathlib import Path
import cv2

# ===== Settings (keep small for first test) =====
KITTI_ROOT = os.path.expanduser("~/kitti_project")
SEQ_LIMIT = 3          # convert first 3 sequences only (0000,0001,0002)
FRAME_LIMIT = 200      # first 200 frames each (set None for all later)
# ===============================================

# YOLO classes: 0=car, 1=person, 2=cyclist
CLS_MAP = {
    "Car": 0,
    "Van": 0,
    "Truck": 0,
    "Pedestrian": 1,
    "Person_sitting": 1,
    "Cyclist": 2,
}

def to_yolo_line(cls_id, x1, y1, x2, y2, w, h):
    # clamp
    x1 = max(0.0, min(x1, w - 1))
    x2 = max(0.0, min(x2, w - 1))
    y1 = max(0.0, min(y1, h - 1))
    y2 = max(0.0, min(y2, h - 1))

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    return f"{cls_id} {cx/w:.6f} {cy/h:.6f} {bw/w:.6f} {bh/h:.6f}"

def main():
    train_dir = Path(KITTI_ROOT) / "training"
    img_root = train_dir / "image_02"
    lbl_root = train_dir / "label_02"

    out_root = Path(KITTI_ROOT) / "yolo_kitti"
    out_img = out_root / "images/train"
    out_lbl = out_root / "labels/train"
    out_meta = out_root / "meta"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    # Save track_id info for later tracking evaluation
    meta_path = out_meta / "tracks.csv"
    with open(meta_path, "w") as meta:
        meta.write("img_id,seq,frame,track_id,class,x1,y1,x2,y2\n")

        seq_dirs = sorted([p for p in img_root.iterdir() if p.is_dir()])[:SEQ_LIMIT]
        print(f"Using sequences: {[p.name for p in seq_dirs]}")

        for seq_dir in seq_dirs:
            seq = seq_dir.name
            label_file = lbl_root / f"{seq}.txt"
            if not label_file.exists():
                print(f"[WARN] Missing label: {label_file}")
                continue

            # Index labels by frame
            frame_map = {}
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 10:
                        continue
                    frame = int(parts[0])
                    track_id = int(parts[1])
                    cls = parts[2]
                    if cls not in CLS_MAP:
                        continue
                    x1, y1, x2, y2 = map(float, parts[6:10])
                    frame_map.setdefault(frame, []).append((track_id, cls, CLS_MAP[cls], x1, y1, x2, y2))

            img_files = sorted(seq_dir.glob("*.png"))
            if FRAME_LIMIT is not None:
                img_files = img_files[:FRAME_LIMIT]

            converted = 0
            for img_path in img_files:
                frame = int(img_path.stem)
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]

                img_id = f"{seq}_{frame:06d}"
                # Save image as jpg to reduce storage
                out_img_path = out_img / f"{img_id}.jpg"
                cv2.imwrite(str(out_img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                anns = frame_map.get(frame, [])
                out_lbl_path = out_lbl / f"{img_id}.txt"
                with open(out_lbl_path, "w") as lf:
                    for (tid, cls, cid, x1, y1, x2, y2) in anns:
                        lf.write(to_yolo_line(cid, x1, y1, x2, y2, w, h) + "\n")
                        meta.write(f"{img_id},{seq},{frame},{tid},{cls},{x1},{y1},{x2},{y2}\n")

                converted += 1

            print(f"Converted seq {seq}: {converted} frames")

    print(f"DONE. Track meta saved at: {meta_path}")
    print("Next: run the script to generate YOLO dataset.")

if __name__ == "__main__":
    main()
