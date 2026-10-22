import torch
import os
import os.path
import re
import numpy as np
from collections import OrderedDict
from lib.train.data import jpeg4py_loader
from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings


def parse_uav123_config(config_file):
    """
    Parse file configSeqs.m cua UAV123 de trich xuat chinh xac 123 sequences duoc dinh nghia
    trong bien `seqUAV123 = {...}`.
    Moi sequence dinh nghia:
      - name: ten sequence (vd: 'bike1', 'bird1_1', 'bird1_2', 'car1_s')
      - folder: thu muc anh thuc te (vd: 'bird1' chua frame cua ca bird1_1 va bird1_2)
      - startFrame: frame bat dau (1-based, vd: 775 cho bird1_2)
      - endFrame: frame ket thuc (1-based, vd: 1477 cho bird1_2)

    Complexity: Time O(L) voi L la so dong trong file (~150 dong), Space O(1).
    """
    if not os.path.isfile(config_file):
        return []
    with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    start_pos = content.find("seqUAV123")
    if start_pos == -1:
        return []
    end_pos = content.find("seqs =", start_pos)
    if end_pos == -1:
        end_pos = len(content)
    block = content[start_pos:end_pos]

    pattern = re.compile(
        r"struct\s*\(\s*'name'\s*,\s*'([^']+)'\s*,\s*'path'\s*,\s*'([^']+)'\s*,\s*'startFrame'\s*,\s*(\d+)\s*,\s*'endFrame'\s*,\s*(\d+)"
    )
    matches = pattern.findall(block)
    records = []
    for name, raw_path, start, end in matches:
        folder = raw_path.replace('\\', '/').strip('/ ').split('/')[-1]
        records.append({
            'name': name,
            'folder': folder,
            'start': int(start),
            'end': int(end)
        })
    return records


def list_sequences(root):
    """ Lists all the video sequence folder names in root """
    if not os.path.isdir(root):
        return []
    sequence_list = []
    for s in os.listdir(root):
        if os.path.isdir(os.path.join(root, s)):
            sequence_list.append(s)
    return sorted(sequence_list)


class UAV123(BaseVideoDataset):
    """ UAV123 dataset for training """
    def __init__(self, root=None, image_loader=jpeg4py_loader):
        root = env_settings().uav123_dir if root is None else root
        super().__init__('UAV123', root, image_loader)

        # Tim thu muc chua anh: thu data_seq/UAV123 truoc, sau do den root
        seq_dir = os.path.join(self.root, "data_seq", "UAV123")
        if not os.path.isdir(seq_dir):
            seq_dir = os.path.join(self.root, "data_seq")
            if not os.path.isdir(seq_dir):
                seq_dir = self.root
        self.seq_dir = seq_dir

        # Doc configSeqs.m de map day du 123 scenarios vao 91 folders
        config_file = os.path.join(self.root, "configSeqs.m")
        self.seq_records = parse_uav123_config(config_file)

        if len(self.seq_records) == 123:
            self.sequence_list = [r['name'] for r in self.seq_records]
            self.seq_record_dict = {r['name']: r for r in self.seq_records}
            print(f"UAV123: 123 sequences mapped from configSeqs.m (images in: {self.seq_dir})")
        else:
            self.sequence_list = list_sequences(self.seq_dir)
            self.seq_record_dict = {}
            print(f"UAV123: {len(self.sequence_list)} folders found in {self.seq_dir} (fallback)")

        self.seq_to_class_map, self.seq_per_class = self._load_class_info()

    def _load_class_info(self):
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        class_map_path = os.path.join(ltr_path, 'data_specs', 'mapping.txt')

        seq_to_class_map = {}
        if os.path.isfile(class_map_path):
            with open(class_map_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        seq_to_class_map[parts[0]] = parts[1]

        seq_per_class = {}
        for i, seq in enumerate(self.sequence_list):
            class_name = seq_to_class_map.get(seq, 'uav')
            if class_name not in seq_per_class:
                seq_per_class[class_name] = [i]
            else:
                seq_per_class[class_name].append(i)

        return seq_to_class_map, seq_per_class

    def get_name(self):
        return "UAV123"

    def has_class_info(self):
        return True

    def _read_bb_anno(self, seq_id):
        video_name = self.sequence_list[seq_id]

        # Check anno/UAV123, then anno/, then root/
        anno_candidates = [
            os.path.join(self.root, "anno", "UAV123", video_name + ".txt"),
            os.path.join(self.root, "anno", video_name + ".txt"),
            os.path.join(self.root, "anno", "UAV123", video_name + "_1.txt"),
            os.path.join(self.root, video_name + ".txt")
        ]

        anno_file = None
        for cand in anno_candidates:
            if os.path.isfile(cand):
                anno_file = cand
                break

        if anno_file is None:
            raise FileNotFoundError(f"Annotation file not found for UAV123 sequence '{video_name}' under {self.root}")

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

        if video_name in self.seq_record_dict:
            rec = self.seq_record_dict[video_name]
            folder = rec['folder']
            # frame_id la 0-indexed trong doan video, can cong startFrame (1-based)
            actual_frame = rec['start'] + frame_id
            frame_id_str = f"{actual_frame:06d}.jpg"

            frame_candidates = [
                os.path.join(self.seq_dir, folder, frame_id_str),
                os.path.join(self.root, "data_seq", "UAV123", folder, frame_id_str),
                os.path.join(self.root, "data_seq", folder, frame_id_str),
            ]
            for cand in frame_candidates:
                if os.path.isfile(cand):
                    return self.image_loader(cand)
            raise FileNotFoundError(f"Frame {actual_frame} not found for UAV123 sequence '{video_name}' in folder '{folder}'")
        else:
            # Fallback 1-based indexing cho thu muc phang
            frame_id_str = str(frame_id + 1).zfill(6) + ".jpg"
            frame_candidates = [
                os.path.join(self.seq_dir, video_name, frame_id_str),
                os.path.join(self.root, "data_seq", "UAV123", video_name, frame_id_str),
            ]
            for cand in frame_candidates:
                if os.path.isfile(cand):
                    return self.image_loader(cand)
            raise FileNotFoundError(f"Frame {frame_id} not found for UAV123 sequence '{video_name}' under {self.seq_dir}")

    def get_class_name(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return self.seq_to_class_map.get(seq_name, 'uav')

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
