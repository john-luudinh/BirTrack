import torch
import os
import numpy as np
from collections import OrderedDict
from lib.train.data import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings


class ANTI300(BaseVideoDataset):
    """ ANTI_300 dataset wrapper for training """
    def __init__(self, root=None, image_loader=jpeg4py_loader, split='train'):
        root = env_settings().anti300_dir if root is None else root
        super().__init__('ANTI_300', root, image_loader)

        self.split = split
        self.dataset_dir = os.path.join(root, split)
        self.anno_dir = os.path.join(root, 'annotations', split)

        # Support fallback between valid and val naming
        if not os.path.isdir(self.dataset_dir):
            alt_split = "val" if split == "valid" else ("valid" if split == "val" else None)
            if alt_split and os.path.isdir(os.path.join(root, alt_split)):
                print(f"[ANTI300] '{split}' directory not found, using '{alt_split}' directory in {root}")
                self.dataset_dir = os.path.join(root, alt_split)
                self.anno_dir = os.path.join(root, 'annotations', alt_split)

        # List all sequences in the split directory
        self.sequence_list = self._get_sequence_list()
        print(f"ANTI_300 ({self.split}): {len(self.sequence_list)} sequences")
        if len(self.sequence_list) == 0:
            print(f"[ANTI300] WARNING: 0 sequences found in {self.dataset_dir}! Check dataset directory and structure.")

    def _get_sequence_list(self):
        if not os.path.isdir(self.dataset_dir):
            return []
        sequence_list = [d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d))]
        return sorted(sequence_list)

    def get_name(self):
        return f"ANTI_300_{self.split}"

    def has_class_info(self):
        return False

    def _read_bb_anno(self, seq_id):
        video_name = self.sequence_list[seq_id]
        anno_file = os.path.join(self.anno_dir, video_name + ".txt")

        if not os.path.isfile(anno_file):
            raise FileNotFoundError("Annotation file not found: {}".format(anno_file))

        with open(anno_file, "r") as f:
            lines = f.readlines()
            bb_anno = []
            for line in lines:
                line = line.strip()
                if line:
                    delimiter = ',' if ',' in line else ' '
                    coords = list(map(float, line.split(delimiter)))
                    bb_anno.append(coords[:4])
        return torch.tensor(bb_anno, dtype=torch.float32)

    def get_sequence_info(self, seq_id):
        bbox = self._read_bb_anno(seq_id)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame(self, seq_id, frame_id):
        video_name = self.sequence_list[seq_id]
        frame_id_str = str(frame_id + 1).zfill(6) # 000001.jpg format (1-indexed)
        frame_path = os.path.join(self.dataset_dir, video_name, frame_id_str + ".jpg")

        if not os.path.isfile(frame_path):
            raise FileNotFoundError("Frame file not found: {}".format(frame_path))
        return self.image_loader(frame_path)

    def get_class_name(self, seq_id):
        return 'uav' # Fixed class for UAV datasets

    def get_frames(self, seq_id, frame_ids, anno=None):
        frame_list = [self._get_frame(seq_id, f) for f in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': self.get_class_name(seq_id),
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
