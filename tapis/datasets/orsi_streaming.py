#!/usr/bin/env python3
"""
Streaming dataset for sequential video analysis.
Processes videos frame by frame in order (streaming fashion) instead of random access.
"""

import os
import sys
from pathlib import Path

_repo_root = Path(__file__).parent.parent.parent
if __name__ == "__main__":
    os.chdir(str(_repo_root))
for _p in [
    str(_repo_root),
    str(_repo_root / "tapis"),
    str(_repo_root / "region_proposals"),
    str(_repo_root / "detectron2"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .build import DATASET_REGISTRY
from .orsi import Orsi


@DATASET_REGISTRY.register()
class Orsi_streaming(Dataset):
    """
    Streaming-style dataset that processes videos sequentially.
    Unlike Orsi which uses random access by clip index, this dataset:
    - Iterates through frames in sequential order
    - Groups frames by video for streaming inference
    - Maintains video context across sequential frames

    For streaming inference, frames are processed in order within each video.
    """

    def __init__(self, cfg, split="test", load=True):
        self.cfg = cfg
        self._split = split
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = cfg.DATA.NUM_FRAMES
        self._seq_mode = cfg.DATA.SEQ_MODE
        self._num_classes = {
            key: n_class for key, n_class in zip(cfg.TASKS.TASKS, cfg.TASKS.NUM_CLASSES)
        }

        # Paths
        self.video_root = getattr(
            cfg.ENDOVIS_DATASET, "ORSI_ROOT_DIR", cfg.ENDOVIS_DATASET.FRAME_DIR
        )
        self.label_dir = getattr(
            cfg.ENDOVIS_DATASET, "ORSI_LABEL_DIR", cfg.ENDOVIS_DATASET.ANNOTATION_DIR
        )
        self.frame_folder = getattr(cfg.ENDOVIS_DATASET, "ORSI_FRAME_FOLDER", "Video_1fps")
        self.label_folder = getattr(cfg.ENDOVIS_DATASET, "ORSI_LABEL_FOLDER", "Label")
        self.image_type = getattr(cfg.ENDOVIS_DATASET, "ORSI_IMAGE_TYPE", "pt")

        # Store loaded data
        self.patient_frame_ids = {}
        self.frame_info = {}
        self.video_frame_lists = {}  # patient -> list of (frame_id, frame_path)
        self.frame_to_label = {}  # (patient, frame_id) -> label info

        if load:
            self._load_data()

    def _list_patient_ids(self):
        if self._split == "train":
            list_files = self.cfg.ENDOVIS_DATASET.TRAIN_LISTS
        elif self._split == "val":
            list_files = self.cfg.ENDOVIS_DATASET.VAL_LISTS
        elif self._split == "test":
            list_files = self.cfg.ENDOVIS_DATASET.TEST_LISTS
        else:
            raise ValueError(f"Unsupported split {self._split}")

        if isinstance(list_files, str):
            list_files = [list_files]

        patients = []
        for f in list_files:
            base = os.path.basename(f)
            if base.endswith(".csv"):
                patient = base[:-4]
                patient = patient.removesuffix("_all_label")
            elif base.endswith(".json"):
                patient = base.replace("_coco.json", "")
            else:
                patient = base
            patients.append(patient)
        return patients

    def _locate_label_file(self, patient):
        candidates = []
        if self.label_dir:
            candidates += [
                os.path.join(
                    self.video_root, patient, self.label_folder, f"{patient}_all_labels.csv"
                )
            ]
        if self.video_root:
            candidates += [
                os.path.join(
                    self.video_root, patient, self.label_folder, f"{patient}_all_labels.csv"
                )
            ]
        for path in candidates:
            if path and os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"Unable to locate label CSV for patient '{patient}'. Searched: {candidates}"
        )

    def _load_data(self):
        """Load data and organize frames sequentially by video."""
        for patient in self._list_patient_ids():
            label_path = self._locate_label_file(patient)
            df = pd.read_csv(label_path)

            # Store frame info for this patient
            frame_list = []
            for _, row in df.iterrows():
                frame_id = row["frame_id"]
                frame_path = row["frame_path"]
                frame_list.append((frame_id, frame_path))

                # Store label info for quick lookup
                self.frame_to_label[(patient, frame_id)] = {
                    "event_id": row.get("event_id", -1),
                    "event_name": row.get("event_name", "Idle"),
                    "phase_id": row.get("phase_id", -1),
                    "phase_name": row.get("phase_name", "Idle"),
                }

            # Sort by frame_id to ensure sequential order
            frame_list.sort(key=lambda x: x[0])
            self.video_frame_lists[patient] = frame_list
            self.patient_frame_ids[patient] = [f[0] for f in frame_list]

    def __len__(self):
        """Return total number of clips across all videos."""
        total = 0
        for patient, frames in self.video_frame_lists.items():
            num_frames = len(frames)
            if num_frames >= self._video_length:
                total += num_frames - self._video_length + 1
        return total

    def get_video_info(self):
        """Return list of (patient, start_frame_idx, num_frames) for streaming."""
        video_info = []
        for patient, frames in self.video_frame_lists.items():
            num_frames = len(frames)
            video_info.append((patient, 0, num_frames))
        return video_info

    def _get_clip_for_frame(self, patient, center_idx):
        """Get a clip centered around frame at center_idx."""
        frames = self.video_frame_lists[patient]
        seq_len = self._video_length * self._sample_rate

        # Calculate sequence indices
        if self._seq_mode == "center":
            start = max(0, center_idx - seq_len // 2)
            end = min(len(frames), start + seq_len)
            seq = list(range(start, end))
        else:
            start = center_idx
            end = min(len(frames), start + seq_len)
            seq = list(range(start, end))

        return seq

    def __getitem__(self, idx):
        """
        For streaming, we iterate through frames sequentially.
        idx corresponds to a sequential position in the streaming order.
        """
        # Find which video and frame this idx corresponds to
        cumulative = 0
        for patient, frames in self.video_frame_lists.items():
            num_frames = len(frames)
            if num_frames < self._video_length:
                continue

            clips_for_video = num_frames - self._video_length + 1
            if idx < cumulative + clips_for_video:
                local_idx = idx - cumulative
                frame_idx = local_idx + self._video_length // 2
                return self._get_item_for_frame(patient, frame_idx)
            cumulative += clips_for_video

        raise IndexError(f"Index {idx} out of range")

    def _get_item_for_frame(self, patient, frame_idx):
        """Get a single item for streaming inference at the given frame."""
        frames = self.video_frame_lists[patient]
        seq = self._get_clip_for_frame(patient, frame_idx)

        # Load frames
        image_paths = [frames[i][1] for i in seq]
        imgs = []
        for path in image_paths:
            full_path = os.path.join(self.video_root, path)
            full_path = full_path.replace("/orsi_tensors/orsi_tensors", "/orsi_tensors/")
            if self.image_type == "pt":
                tensor = torch.load(full_path, map_location="cpu", weights_only=True)
                if isinstance(tensor, torch.Tensor):
                    imgs.append(tensor.numpy())
                else:
                    imgs.append(tensor)
            else:
                import cv2

                img = cv2.imread(full_path)
                imgs.append(img)

        # Process images
        if isinstance(imgs[0], np.ndarray):
            imgs = np.stack(imgs)
            imgs = torch.from_numpy(imgs).float()
            if imgs.max() > 10.0:
                imgs = imgs / 255.0
            if imgs.ndim == 4 and imgs.shape[-1] == 3:
                imgs = imgs.permute(3, 0, 1, 2)  # NCHW -> CTHW
            elif imgs.ndim == 4 and imgs.shape[1] == 3:
                imgs = imgs.permute(1, 0, 2, 3)  # NCHW -> CTHW

        # Get labels for center frame
        center_frame_id = frames[frame_idx][0]
        label_info = self.frame_to_label.get((patient, center_frame_id), {})

        all_labels = {}
        for task in self._num_classes:
            if task == "steps":
                all_labels[task] = label_info.get("event_id", -1)
            elif task == "phases":
                all_labels[task] = label_info.get("phase_id", -1)

        frame_identifier = f"{patient}/{center_frame_id:09d}.{self.image_type}"

        return [imgs], all_labels, {}, frame_identifier


if __name__ == "__main__":
    import argparse
    from fvcore.common.config import CfgNode
    from tapis.config.defaults import assert_and_infer_cfg
    from tapis.config.defaults import get_cfg as _get_cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    cfg = _get_cfg()
    cfg.merge_from_file(args.cfg)
    cfg = assert_and_infer_cfg(cfg)
    cfg.freeze()

    dataset = OrsiStreaming(cfg, split=args.split)
    print(f"Total streaming clips: {len(dataset)}")
    print(f"Videos: {list(dataset.video_frame_lists.keys())}")
