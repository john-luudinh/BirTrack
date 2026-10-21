"""
Preprocess UAV-ANTI-UAV raw dataset.

Target structure for SGLATrack (similar to Data/ folder):
UAV_ANTI_UAV_processed/
├── train/
│   └── <seq_name>/
│       ├── 000001.jpg ...
├── test/
│   └── <seq_name>/
│       ├── 000001.jpg ...
└── annotations/
    ├── train/
    │   └── <seq_name>.txt
    └── test/
        └── <seq_name>.txt
"""

import os
import cv2
import argparse
import shutil


def extract_frames(video_path: str, output_dir: str, frame_format: str = "%06d.jpg") -> int:
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


def preprocess_uav_anti_uav(raw_dir: str, output_dir: str = None) -> None:
    """Main preprocessing pipeline for UAV-ANTI-UAV dataset.

    Args:
        raw_dir: Path to the original UAV-ANTI-UAV folder containing Train/ and Test/.
        output_dir: Path for UAV_ANTI_UAV_processed. Defaults to <parent of raw_dir>/UAV_ANTI_UAV_processed.
    """
    raw_dir = os.path.abspath(raw_dir)
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(raw_dir), "UAV_ANTI_UAV_processed")

    # Map raw split names to processed split names
    splits_mapping = {
        "Train": "train",
        "Test": "test"
    }

    for raw_split, out_split in splits_mapping.items():
        split_dir = os.path.join(raw_dir, raw_split)
        if not os.path.isdir(split_dir):
            print(f"[WARN] Split directory not found, skipping: {split_dir}")
            continue

        seq_names = sorted([
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        ])
        print(f"\n=== Processing split '{raw_split}': {len(seq_names)} sequences ===")

        for seq_idx, seq_name in enumerate(seq_names, start=1):
            seq_dir = os.path.join(split_dir, seq_name)
            
            # --- Extract frames ---
            video_file = seq_name + ".mp4"
            video_path = os.path.join(seq_dir, video_file)
            frames_out_dir = os.path.join(output_dir, out_split, seq_name)

            if not os.path.isfile(video_path):
                print(f"  [WARN] Video not found: {video_path}")
                continue

            # Skip if frames already extracted
            if os.path.isdir(frames_out_dir) and len(os.listdir(frames_out_dir)) > 0:
                n_existing = len([f for f in os.listdir(frames_out_dir) if f.endswith(".jpg")])
                print(f"  [{seq_idx}/{len(seq_names)}] {seq_name} - already extracted ({n_existing} frames), skipping.")
            else:
                n_frames = extract_frames(video_path, frames_out_dir)
                print(f"  [{seq_idx}/{len(seq_names)}] {seq_name} - extracted {n_frames} frames")

            # --- Copy annotation ---
            # groundtruth_rect.txt already contains 0,0,0,0 for absent frames, so direct copy is fine
            gt_path = os.path.join(seq_dir, "groundtruth_rect.txt")
            anno_out_dir = os.path.join(output_dir, "annotations", out_split)
            anno_txt_path = os.path.join(anno_out_dir, seq_name + ".txt")

            os.makedirs(anno_out_dir, exist_ok=True)

            if not os.path.isfile(gt_path):
                print(f"  [WARN] Annotation groundtruth_rect.txt not found: {gt_path}")
                continue

            shutil.copy2(gt_path, anno_txt_path)

    print(f"\n[DONE] Processed UAV-ANTI-UAV dataset saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess UAV-ANTI-UAV raw dataset (MP4) into frame-extracted "
                    "format compatible with SGLATrack Data/ convention."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Path to the original UAV-ANTI-UAV directory (containing Train/ and Test/)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path for the output UAV_ANTI_UAV_processed directory. "
             "Defaults to <parent of raw_dir>/UAV_ANTI_UAV_processed."
    )
    args = parser.parse_args()

    preprocess_uav_anti_uav(args.raw_dir, args.output_dir)
