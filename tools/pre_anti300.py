"""
This script was used to format anti300 dataset
Information
    - From original dataset like was described in report_dataset.md
    - Sample format dataset for tracking in folder Data
Target:
    - Create new dataset folder anti300_processed include in same folder with anti_300, inside has 2 folder:
        + anti300_ir:
        + anti300_rgb:
    - anti300_ir and anti300_rgb has same construct with folder Data (has annotations and train or valid or test outside)
"""

import os
import json
import cv2
import argparse


def extract_frames(video_path, output_dir, frame_format="%06d.jpg"):
    """Extract all frames from an MP4 video and save as JPEG images.

    Args:
        video_path: Path to the .mp4 file.
        output_dir: Directory to write extracted frame images.
        frame_format: Naming pattern for output files (default: 000001.jpg).

    Returns:
        Number of frames extracted.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame_path = os.path.join(output_dir, frame_format % frame_idx)
        cv2.imwrite(frame_path, frame)

    cap.release()
    return frame_idx


def convert_annotation(json_path, output_txt_path):
    """Convert an anti300 JSON annotation file to a per-line x,y,w,h text file.

    The JSON contains:
        - "gt_rect": list of [x, y, w, h] per frame
        - "exist":   list of 0/1 per frame (target presence)

    When exist == 0, the bbox is written as 0,0,0,0 so the dataloader can
    mark it as valid=False via (w > 0) & (h > 0).

    Args:
        json_path: Path to the visible.json or infrared.json file.
        output_txt_path: Destination .txt file path.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_rect = data["gt_rect"]
    exist = data.get("exist", [1] * len(gt_rect))

    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        for i, bbox in enumerate(gt_rect):
            if i < len(exist) and exist[i] == 0:
                f.write("0,0,0,0\n")
            else:
                x, y, w, h = bbox
                f.write(f"{x},{y},{w},{h}\n")


def preprocess_anti300(raw_dir, output_dir=None):
    """Main preprocessing pipeline for anti300 dataset.

    Reads the raw anti300 directory structure:
        anti300/
        ├── train/
        │   └── <seq_name>/
        │       ├── visible.mp4
        │       ├── infrared.mp4
        │       ├── visible.json
        │       └── infrared.json
        ├── val/
        └── test/

    Produces the processed output:
        anti300_processed/
        ├── anti300_rgb/
        │   ├── train/
        │   │   └── <seq_name>/
        │   │       ├── 000001.jpg ...
        │   ├── val/
        │   ├── test/
        │   └── annotations/
        │       ├── train/
        │       │   └── <seq_name>.txt
        │       ├── val/
        │       └── test/
        └── anti300_ir/
            └── (same structure)

    Args:
        raw_dir: Path to the original anti300 folder containing train/val/test.
        output_dir: Path for anti300_processed. Defaults to <parent of raw_dir>/anti300_processed.
    """
    raw_dir = os.path.abspath(raw_dir)
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(raw_dir), "anti300_processed")

    # Channel mapping: (subfolder name, video filename, json filename)
    channels = [
        ("anti300_rgb", "visible.mp4", "visible.json"),
        ("anti300_ir", "infrared.mp4", "infrared.json"),
    ]

    splits = ["train", "val", "test"]

    for split in splits:
        split_dir = os.path.join(raw_dir, split)
        if not os.path.isdir(split_dir):
            print(f"[WARN] Split directory not found, skipping: {split_dir}")
            continue

        # Collect all sequence directories in this split
        seq_names = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])
        print(f"\n=== Processing split '{split}': {len(seq_names)} sequences ===")

        for seq_idx, seq_name in enumerate(seq_names, start=1):
            seq_dir = os.path.join(split_dir, seq_name)

            for channel_folder, video_file, json_file in channels:
                # --- Extract frames ---
                video_path = os.path.join(seq_dir, video_file)
                frames_out_dir = os.path.join(output_dir, channel_folder, split, seq_name)

                if not os.path.isfile(video_path):
                    print(f"  [WARN] Video not found: {video_path}")
                    continue

                # Skip if frames already extracted
                if os.path.isdir(frames_out_dir) and len(os.listdir(frames_out_dir)) > 0:
                    n_existing = len([
                        f for f in os.listdir(frames_out_dir) if f.endswith(".jpg")
                    ])
                    print(f"  [{seq_idx}/{len(seq_names)}] {seq_name}/{channel_folder} "
                          f"- already extracted ({n_existing} frames), skipping.")
                else:
                    n_frames = extract_frames(video_path, frames_out_dir)
                    print(f"  [{seq_idx}/{len(seq_names)}] {seq_name}/{channel_folder} "
                          f"- extracted {n_frames} frames")

                # --- Convert annotation ---
                json_path = os.path.join(seq_dir, json_file)
                anno_txt_path = os.path.join(
                    output_dir, channel_folder, "annotations", split, seq_name + ".txt"
                )

                if not os.path.isfile(json_path):
                    print(f"  [WARN] Annotation JSON not found: {json_path}")
                    continue

                convert_annotation(json_path, anno_txt_path)

    print(f"\n[DONE] Processed dataset saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess anti300 raw dataset (MP4+JSON) into frame-extracted "
                    "format compatible with SGLATrack Data/ convention."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Path to the original anti300 directory (containing train/, val/, test/)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path for the output anti300_processed directory. "
             "Defaults to <parent of raw_dir>/anti300_processed."
    )
    args = parser.parse_args()

    preprocess_anti300(args.raw_dir, args.output_dir)

# python scripts/pre_anti300.py --raw_dir /mnt/DATA/tuld3/Datasets/tracking/anti300