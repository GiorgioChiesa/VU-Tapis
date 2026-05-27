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
        self.exclude_event_names = {
            name.strip().lower() for name in getattr(cfg.ENDOVIS_DATASET, "EXCLUDE_EVENT_NAMES", [])
        }

        # Store loaded data
        self.patient_frame_ids = {}
        self.frame_info = {}
        self.video_frame_lists = {}  # patient -> list of (frame_id, frame_path)
        self.frame_to_label = {}  # (patient, frame_id) -> label info
        self.dfs = pd.DataFrame()
        self.filtered_dfs = None
        self.counter = {}

        # Load mappings from events_lists
        if self.video_root not in sys.path:
            sys.path.insert(0, self.video_root)
        try:
            from events_lists import mapping_events_name_to_id, mapping_phases_name_to_id, class_mapping, classes
            self.classes_eventname2class_name = class_mapping
            self.classes_name2idx = classes
            self.classes_idx2name = {v: k for k, v in classes.items()}
            self.event_name2idx = mapping_events_name_to_id
            self.phase_name2idx = mapping_phases_name_to_id
            self.event_idx2name = {v: k for k, v in mapping_events_name_to_id.items()}
            self.phase_idx2name = {v: k for k, v in mapping_phases_name_to_id.items()}
        except ImportError as e:
            print(f"Error importing events_lists: {e}")
            self.event_name2idx = {}
            self.phase_name2idx = {}
            self.classes_idx2name = {}
            self.event_idx2name = {}
            self.phase_idx2name = {}

        if load:
            self._load_data()
            if any(self.cfg.TASKS.WEIGHT_LOSS_BY_CLASS):
                self.generate_weight_vector()
                self.remapping_local_id()

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
        dfs = []
        
        for patient in self._list_patient_ids():
            label_path = self._locate_label_file(patient)
            df = pd.read_csv(label_path)
            dfs.append(df)

            # Store frame info for this patient
            frame_list = []
            for _, row in df.iterrows():
                frame_id = row["frame_id"]
                frame_path = row["frame_path"]
                frame_list.append((frame_id, frame_path))

                # Store label info for quick lookup
                self.frame_to_label[(patient, frame_id)] = {
                    "event_id": row.get("event_id", -1),
                    "event_name": row.get("event_name", "Unknown"),
                    "phase_id": row.get("phase_id", -1),
                    "phase_name": row.get("phase_name", "Unknown"),
                    "classes_id": row.get("classes_id", -1),
                    "classes_name": row.get("classes_name", "Unknown"),
                }

            # Sort by frame_id to ensure sequential order
            frame_list.sort(key=lambda x: x[0])
            self.video_frame_lists[patient] = frame_list
            self.patient_frame_ids[patient] = [f[0] for f in frame_list]

        # Concatenate all dataframes
        self.dfs = pd.concat(dfs, ignore_index=True)

        # Filter based on exclude_event_names
        if self.exclude_event_names:
            exclude = self.exclude_event_names
            
            exclude.discard("idle")
            self.filtered_dfs = self.dfs[
                ~self.dfs["event_name"].str.strip().str.lower().str.replace(" ", "_").isin(exclude)
            ].reset_index(drop=True)
        else:
            self.filtered_dfs = self.dfs.copy()

        # # Remove idle frames if ADD_IDLE is enabled
        # if getattr(self.cfg.ENDOVIS_DATASET, "ADD_IDLE", 0) > 0:
        #     self.filtered_dfs = self.filtered_dfs[
        #         self.filtered_dfs["event_name"].str.strip().str.lower() != "idle"
        #     ].reset_index(drop=True)
        
        self.collapse_event_dfs()


        # Ensure index is properly reset
        self.filtered_dfs = self.filtered_dfs.reset_index(drop=True)

        # Save the dataset CSV
        saving = self.filtered_dfs if self.filtered_dfs is not None else self.dfs
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        saving.to_csv(os.path.join(self.cfg.OUTPUT_DIR, f"{self._split}_data.csv"), index=False)


    def collapse_event_dfs(self):
        """
        Collapse multiple events into one by mapping certain event names to others.

        Mapping rules:
        - Instrument_swap:_removal -> Removal_of_robotic_instruments
        - Hemolock_clip_on_right_pedicle -> Metal_clip_on_right_pedicle
        - Hemolock_clip_on_left_pedicle -> Metal_clip_on_left_pedicle
        - Events with 'left'/'right' in name are collapsed to the same event
        - Insert_gauze -> Insert_hemostatic_agens
        - Cutting_the_needles -> Removing_needles
          (e.g., 'Clip_on_left_pedicle' and 'Clip_on_right_pedicle' become the same)
        """
        dfs = self.filtered_dfs if self.filtered_dfs is not None else self.dfs

        # Define explicit collapse mappings: source_event -> target_event
        collapse_mapping = {
            "instrument_swap:_removal": "removal_of_robotic_instruments",
            "hemolock_clip_on_right_pedicle": "metal_clip_on_right_pedicle",
            "hemolock_clip_on_left_pedicle": "metal_clip_on_left_pedicle",
            "insert_gauze": "insert_hemostatic_agens",
            "cutting_the_needles": "removing_needles",
        }

        # Get unique events from the dataframe
        unique_events = dfs[["event_id", "event_name"]].drop_duplicates()

        # Create a lookup for event IDs by normalized name
        event_lookup = {}
        for _, row in unique_events.iterrows():
            normalized_name = str(row["event_name"]).strip().lower().replace(" ", "_")
            event_lookup[normalized_name] = {
                "event_id": row["event_id"],
                "event_name": row["event_name"],
            }

        # Apply explicit collapse mapping
        for source_name, target_name in collapse_mapping.items():
            if target_name in event_lookup:
                target_event = event_lookup[target_name]
                self.exclude_event_names.add(source_name)
                mask = (
                    dfs["event_name"].str.strip().str.lower().str.replace(" ", "_") == source_name
                )
                if mask.any():
                    dfs.loc[mask, "event_id"] = target_event["event_id"]
                    dfs.loc[mask, "event_name"] = target_event["event_name"]
                    print(f"Collapsed '{source_name}' -> '{target_name}' ({mask.sum()} rows)")
            else:
                print(f"Warning: Target event '{target_name}' not found in dataset")

        # Collapse events that differ only by left/right
        # Group events by base name (without left/right)
        left_right_groups = {}
        for _, row in unique_events.iterrows():
            normalized_name = str(row["event_name"]).strip().lower().replace(" ", "_")
            # Remove left/right prefixes/suffixes to get base name
            base_name = normalized_name.replace("_left", "").replace("_right", "")
            base_name = base_name.replace("left_", "").replace("right_", "")

            if base_name not in left_right_groups:
                left_right_groups[base_name] = []
            left_right_groups[base_name].append(
                {
                    "original_name": normalized_name,
                    "event_id": row["event_id"],
                    "event_name": row["event_name"],
                }
            )

        # For each group with multiple variants, collapse to one (prefer "right" or lower ID)
        for base_name, variants in left_right_groups.items():
            if len(variants) > 1:
                # Check if variants actually differ by left/right
                has_left = any("left" in v["original_name"] for v in variants)
                has_right = any("right" in v["original_name"] for v in variants)

                if has_left and has_right:
                    # Prefer "right" variant, otherwise use lowest ID
                    target = next(
                        (v for v in variants if "right" in v["original_name"]),
                        min(variants, key=lambda x: x["event_id"]),
                    )

                    # Collapse all variants to target
                    for variant in variants:
                        if variant["event_id"] != target["event_id"]:
                            self.exclude_event_names.add(variant["original_name"])
                            mask = (
                                dfs["event_name"].str.strip().str.lower().str.replace(" ", "_")
                                == variant["original_name"]
                            )
                            if mask.any():
                                dfs.loc[mask, "event_id"] = target["event_id"]
                                dfs.loc[mask, "event_name"] = target["event_name"]
                                print(
                                    f"Collapsed left/right '{variant['original_name']}' -> '{target['original_name']}' ({mask.sum()} rows)"
                                )

        # Update filtered_dfs with the modified dataframe
        if self.filtered_dfs is not None:
            self.filtered_dfs = dfs
        else:
            self.dfs = dfs

    def remapping_local_id(self):
        """Remap class IDs based on distribution counter."""
        map_task = {"steps": "event", "phases": "phase", "actions": "event", "classes": "classes"}
        for task in self.cfg.TASKS.TASKS:
            id_map = dict(zip(self.counter[task]["id"], range(len(self.counter[task]))))
            self.filtered_dfs[f"{map_task[task]}_id"] = (
                self.filtered_dfs[f"{map_task[task]}_id"].map(id_map).fillna(0)
            )

    def generate_weight_vector(self):
        """Generate class distribution statistics and save to CSV."""
        map_task = {"steps": "event", "phases": "phase", "actions": "event", "classes": "classes"}
        self.counter = {}
        for task, weight_loss_by_class in zip(
            self.cfg.TASKS.TASKS, self.cfg.TASKS.WEIGHT_LOSS_BY_CLASS
        ):
            clip = self.filtered_dfs.copy() if self.filtered_dfs is not None else self.dfs.copy()

            his = []
            if task in ["steps", "actions"]:
                mapping = self.event_idx2name
            elif task == "phases":
                mapping = self.phase_idx2name
            elif task == "classes":
                mapping = self.classes_idx2name
            else:
                mapping = {}
                
            for id, event in mapping.items():
                if event.strip().lower().replace(" ", "_") in self.exclude_event_names:
                    continue
                his.append(
                    {
                        "id": id,
                        "name": event,
                        "total_count": sum(clip[f"{map_task[task]}_id"] == id),
                    }
                )
            self.counter[task] = pd.DataFrame(his)

            assert self.counter[task]["total_count"].sum() <= len(clip), (
                f"Total count in distribution ({self.counter[task]['total_count'].sum()}) does not match total samples in dataset ({len(clip)})"
            )
            assert len(self.counter[task]) <= self._num_classes[task], (
                f"Numero di classi non coincide per {task}, deve essere: {len(self.counter[task])}"
            )

            print(f"Weight loss by class for task {task} -- {self._split}:\n{self.counter[task]}")
            if weight_loss_by_class:
                distributions_dir = os.path.join(self.cfg.OUTPUT_DIR, "distributions")
                os.makedirs(distributions_dir, exist_ok=True)
                csv_path = os.path.join(distributions_dir, f"{self._split}_{weight_loss_by_class}")
                self.counter[task].to_csv(csv_path, index=False)

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
            elif task == "classes":
                all_labels[task] = label_info.get("classes_id", -1)

        frame_identifier = f"{patient}/{center_frame_id:09d}.{self.image_type}"

        return [imgs], all_labels, {}, frame_identifier
    
    def frame_name_spliting(self, video_name, sec):
        video_num = int(video_name.replace("RARP", ""))
        return [video_num, sec]

    def frame_num_joining(self, video_num, sec):
        return f"RARP{video_num:03d}/{sec:0{self.zero_fill}d}.{self.image_type}"

    def frame_name_joining(self, video_name, sec):
        return f"{video_name}/IMAGES/{sec:0{self.zero_fill}d}.{self.image_type}"


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
