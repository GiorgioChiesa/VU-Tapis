#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

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

import itertools
import logging
from copy import deepcopy

import numpy as np
import pandas as pd
import regex as re
import torch

from tapis.datasets import cv2_transform

from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class Orsi(torch.utils.data.Dataset):
    """
    Orsi dataloader specific for `data/orsi_tensors/` structure.

    - patient folder: RARPxx
    - frames: RARPxx/Video_1fps/*.pt (1 fps)
    - labels: RARPxx/Label/RARPxx_all_label.csv
    - CSV columns: patient_id, patient_name, frame_id, frame_path, second,
      event_id, event_name, phase_id, phase_name
    - multiple patients are merged in the same dataset
    - filtering by event names through cfg.ENDOVIS_DATASET.EXCLUDE_EVENT_NAMES
    - for a clip, label is nearest non-Idle label relative to central frame.
    - Idle is used only if no non-idle label is available.
    - event labels last 3 seconds (3 frames at 1 fps), with only first frame stored.
    """

    def __init__(self, cfg, split, load=True):
        self.dataset_name = "Orsi"
        self.zero_fill = 9
        self.image_type = "jpg"
        self.cfg = cfg
        self._split = split
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._video_length = (
            max(cfg.DATA.NUM_FRAMES, cfg.MODEL.MEMORY_BANK_SIZE)
            if split == "train"
            else cfg.DATA.NUM_FRAMES
        )
        self._seq_mode = cfg.DATA.SEQ_MODE
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = {
            key: n_class for key, n_class in zip(cfg.TASKS.TASKS, cfg.TASKS.NUM_CLASSES)
        }
        self._region_tasks = {
            task for task in cfg.TASKS.TASKS if task in cfg.ENDOVIS_DATASET.REGION_TASKS
        }
        self._frame_tasks = {
            task for task in cfg.TASKS.TASKS if task not in cfg.ENDOVIS_DATASET.REGION_TASKS
        }

        # Augmentation params.
        self._data_mean = cfg.DATA.MEAN
        self._data_std = cfg.DATA.STD
        self._use_bgr = cfg.ENDOVIS_DATASET.BGR
        self.random_horizontal_flip = cfg.DATA.RANDOM_FLIP
        if self._split == "train":
            self._crop_size = (cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE_LARGE)
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.ENDOVIS_DATASET.TRAIN_USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.ENDOVIS_DATASET.TRAIN_PCA_JITTER_ONLY
            self._pca_eigval = cfg.DATA.TRAIN_PCA_EIGVAL
            self._pca_eigvec = cfg.DATA.TRAIN_PCA_EIGVEC
        else:
            self._crop_size = (cfg.DATA.VAL_CROP_SIZE, cfg.DATA.VAL_CROP_SIZE_LARGE)
            self._test_force_flip = cfg.ENDOVIS_DATASET.VAL_FORCE_FLIP
            self.aspect_ratio_th = cfg.ENDOVIS_DATASET.ASPECT_RATION_TH

        # paths
        self.video_root = getattr(
            cfg.ENDOVIS_DATASET, "ORSI_ROOT_DIR", cfg.ENDOVIS_DATASET.FRAME_DIR
        )
        self.label_dir = getattr(
            cfg.ENDOVIS_DATASET, "ORSI_LABEL_DIR", cfg.ENDOVIS_DATASET.ANNOTATION_DIR
        )
        self.frame_folder = getattr(cfg.ENDOVIS_DATASET, "ORSI_FRAME_FOLDER", "Video_1fps")
        self.frame_folder_alternatives = [self.frame_folder, "IMAGES"]
        self.label_folder = getattr(cfg.ENDOVIS_DATASET, "ORSI_LABEL_FOLDER", "Label")
        self.exclude_event_names = {
            name.strip().lower() for name in getattr(cfg.ENDOVIS_DATASET, "EXCLUDE_EVENT_NAMES", [])
        }
        self.image_type = getattr(cfg.ENDOVIS_DATASET, "ORSI_IMAGE_TYPE", "pt")

        # Store loaded data
        self.patient_frame_ids = {}
        self.patient_frame_paths = {}
        self.frame_info = {}
        self.dfs = pd.DataFrame()
        self.filtered_dfs = None
        self.clips = []

        if self.video_root not in sys.path:
            sys.path.insert(0, self.video_root)
        try:
            from events_lists import mapping_events_name_to_id, mapping_phases_name_to_id

            self.event_name2idx = mapping_events_name_to_id
            self.phase_name2idx = mapping_phases_name_to_id
            self.event_idx2name = {v: k for k, v in mapping_events_name_to_id.items()}
            self.phase_idx2name = {v: k for k, v in mapping_phases_name_to_id.items()}
        except ImportError as e:
            print(f"Error importing events_lists: {e}")
            self.event_name2idx = {}
            self.phase_name2idx = {}
        if load:
            self._load_data()
            if any(self.cfg.TASKS.WEIGHT_LOSS_BY_CLASS):
                self.generate_weight_vector()
                self.remapping_local_id()

    def remapping_local_id(self):
        map_task = {"steps": "event", "phases": "phase"}
        for task in self.cfg.TASKS.TASKS:
            id_map = dict(zip(self.counter[task]["id"], range(len(self.counter[task]))))
            self.filtered_dfs[f"{map_task[task]}_id"] = (
                self.filtered_dfs[f"{map_task[task]}_id"].map(id_map).fillna(0)
            )

    def generate_weight_vector(self):
        map_task = {"steps": "event", "phases": "phase"}
        self.counter = {}
        for task, weight_loss_by_class in zip(
            self.cfg.TASKS.TASKS, self.cfg.TASKS.WEIGHT_LOSS_BY_CLASS
        ):
            clip = self.filtered_dfs.copy() if self.filtered_dfs is not None else self.dfs.copy()

            his = []
            mapping = self.event_idx2name if task == "steps" else self.phase_idx2name
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

            # if "total_count" not in list(clip.columns):
            #     clip.insert(len(clip.columns), "total_count",[0]*len(clip))

            # df = clip.groupby([f"{map_task[task]}_id",f"{map_task[task]}_name"])['total_count'].agg('count').reset_index()
            # print(f"Distribution for task {task} before weight computation:\n{self.counter[task]}")
            # print(df)

            assert self.counter[task]["total_count"].sum() <= len(clip), (
                f"Total count in distribution ({self.counter[task]['total_count'].sum()}) does not match total samples in dataset ({len(clip)})"
            )
            assert len(self.counter[task]) == self._num_classes[task], (
                f"Numero di classi non coincide, deve essere: {len(self.counter[task])}"
            )

            print(f"Weight loss by class for task {task} -- {self._split}:\n{self.counter[task]}")
            if weight_loss_by_class and self._split == "train":
                csv_path = os.path.join(self.cfg.OUTPUT_DIR, "distributions", weight_loss_by_class)
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                self.counter[task].to_csv(csv_path, index=False)

    def _list_patient_ids(self):
        if self._split == "train":
            list_files = self.cfg.ENDOVIS_DATASET.TRAIN_LISTS
        elif self._split == "val":
            list_files = self.cfg.ENDOVIS_DATASET.VAL_LISTS
        elif self._split == "test":
            list_files = self.cfg.ENDOVIS_DATASET.TEST_LISTS
        else:
            raise ValueError(f"Unsupported split {self._split} for Orsi dataset")

        if isinstance(list_files, str):
            list_files = [list_files]

        patients = []
        for f in list_files:
            base = os.path.basename(f)
            if base.endswith(".csv"):
                patient = base[:-4]
                patient = patient.removesuffix("_all_label")
                patients.append(patient)
            else:
                patients.append(base)
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
        # 1) Read labels and expand events over 3 frames

        dfs = []
        for patient in self._list_patient_ids():
            label_path = self._locate_label_file(patient)
            frame_info = {}

            df = pd.read_csv(label_path)
            dfs.append(df)

        self.dfs = pd.concat(dfs, ignore_index=True)

        if self.exclude_event_names:
            exclude = self.exclude_event_names
            if getattr(self.cfg.ENDOVIS_DATASET, "ADD_IDLE", 0) > 0:
                exclude.discard("idle")
            self.filtered_dfs = self.dfs[
                ~self.dfs["event_name"].str.strip().str.lower().str.replace(" ", "_").isin(exclude)
            ].reset_index(drop=True)
        else:
            self.filtered_dfs = self.dfs.copy()

        if getattr(self.cfg.ENDOVIS_DATASET, "ADD_IDLE", 0) > 0:
            self.filtered_dfs = self.filtered_dfs[
                self.filtered_dfs["event_name"].str.strip().str.lower() != "idle"
            ].reset_index(drop=True)

        self._add_idle_frames()

        print(f"DEBUG: After _add_idle_frames - filtered_dfs has {len(self.filtered_dfs)} rows")
        idle_check = self.filtered_dfs[
            self.filtered_dfs["event_name"].str.strip().str.lower() == "idle"
        ]
        print(f"DEBUG: Idle in filtered_dfs: {len(idle_check)}")

        self.collapse_event_dfs()

        saving = self.filtered_dfs if self.filtered_dfs is not None else self.dfs
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

    def _add_idle_frames(self):
        num_idle_frames = getattr(self.cfg.ENDOVIS_DATASET, "ADD_IDLE", 0)
        if num_idle_frames <= 0:
            return
        print(f"Adding up to {num_idle_frames} Idle frames per patient to {self._split} dataset")

        patient_names = self._list_patient_ids()
        if not patient_names:
            return

        all_patient_names = self.dfs["patient_name"].unique()
        name_to_id = dict(
            zip(all_patient_names, (int(re.findall(r"(\d+)", i)[0]) for i in all_patient_names))
        )

        idle_seconds_needed = 30
        idle_frames = []
        selected_frame_ids = set()

        for patient_name in patient_names:
            patient_id = name_to_id.get(patient_name)
            if patient_id is None:
                continue
            patient_df = self.dfs[self.dfs["patient_id"] == patient_id].copy()
            if len(patient_df) == 0:
                continue

            patient_df = patient_df.sort_values("frame_id").reset_index(drop=True)
            idle_rows = (
                patient_df[patient_df["event_name"].str.strip().str.lower() == "idle"]
                .sample(frac=1, random_state=1)
                .reset_index(drop=True)
            )

            if idle_rows.empty:
                continue

            non_idle_events = patient_df[
                patient_df["event_name"].str.strip().str.lower() != "idle"
            ]["frame_id"].values

            valid_idle_rows = []
            for _, row in idle_rows.iterrows():
                frame_id = row["frame_id"]

                too_close = False
                for event_frame_id in non_idle_events:
                    if abs(event_frame_id - frame_id) <= idle_seconds_needed:
                        too_close = True
                        break

                if not too_close:
                    valid_idle_rows.append(row.to_dict())
                if len(valid_idle_rows) >= num_idle_frames:
                    break

            if not valid_idle_rows:
                continue

            # total_frames = len(patient_df)
            # num_bins = min(num_idle_frames, len(valid_idle_rows))
            # bin_size = total_frames / max(num_bins, 1)

            # patient_selected = []
            # for bin_idx in range(num_bins):
            #     start_frame = int(bin_idx * bin_size)
            #     end_frame = int((bin_idx + 1) * bin_size)

            #     candidates = [
            #         r
            #         for r in valid_idle_rows
            #         if r["patient_id"] == patient_id
            #         and start_frame <= r["frame_id"] < end_frame
            #         and r["frame_id"] not in selected_frame_ids
            #     ]

            #     if candidates:
            #         bin_center = (start_frame + end_frame) / 2
            #         best_candidate = min(candidates, key=lambda r: abs(r["frame_id"] - bin_center))
            #         patient_selected.append(best_candidate)
            #         selected_frame_ids.add(best_candidate["frame_id"])

            idle_frames.extend(valid_idle_rows)

        if idle_frames:
            idle_df = pd.DataFrame(idle_frames)
            self.filtered_dfs = pd.concat([self.filtered_dfs, idle_df], ignore_index=True)
            print(f"Added {len(idle_frames)} Idle frames to {self._split} dataset")

    def _select_center_label(self, patient, frame_ids, seq):
        n = len(frame_ids)
        if n == 0:
            return None

        center_idx = seq[len(seq) // 2]

        # Find nearest non-Idle event/phase label.
        best_event = None
        best_phase = None

        for delta in range(0, max(center_idx, n - center_idx - 1) + 1):
            checks = []
            if center_idx - delta >= 0:
                checks.append(center_idx - delta)
            if delta > 0 and center_idx + delta < n:
                checks.append(center_idx + delta)

            for p in checks:
                info = self.frame_info[patient][frame_ids[p]]
                if best_event is None and info.get("event_name", "Idle").strip().lower() != "idle":
                    best_event = info
                if best_phase is None and info.get("phase_name", "Idle").strip().lower() != "idle":
                    best_phase = info
                if best_event is not None and best_phase is not None:
                    break
            if best_event is not None and best_phase is not None:
                break

        if best_event is None:
            best_event = self.frame_info[patient][frame_ids[center_idx]]
        if best_phase is None:
            best_phase = self.frame_info[patient][frame_ids[center_idx]]

        return {
            "event_id": best_event.get("event_id", -1),
            "event_name": best_event.get("event_name", "Idle"),
            "phase_id": best_phase.get("phase_id", -1),
            "phase_name": best_phase.get("phase_name", "Idle"),
        }

    def _label_to_numeric(self, label_id, label_name, name2idx, num_classes):
        if label_id is not None and isinstance(label_id, (int, float)) and label_id > 0:
            out = int(label_id) - 1
            if 0 <= out < num_classes:
                return out
        s = str(label_name).strip().lower()
        if s == "idle" or s == "" or s == "nan":
            return 0
        if s in name2idx:
            out = name2idx[s]
            if out < num_classes:
                return out
        return 0

    def __len__(self):
        return len(self.filtered_dfs) if self.filtered_dfs is not None else len(self.dfs)

    def _images_and_boxes_preprocessing_cv2(self, imgs, boxes=None, image=None):
        height, width, _ = imgs[0].shape
        if boxes is None:
            boxes = np.zeros((1, 4))
        boxes = cv2_transform.clip_boxes_to_image(boxes, height, width)

        boxes = [boxes.astype("float")]

        if self._split == "train" and not self.cfg.DATA.JUST_CENTER:
            imgs, boxes = cv2_transform.random_short_side_scale_jitter_list(
                imgs, min_size=self._jitter_min_scale, max_size=self._jitter_max_scale, boxes=boxes
            )
            imgs, boxes, image = cv2_transform.random_crop_list(
                imgs, self._crop_size, order="HWC", boxes=boxes, image=image
            )

            if self.random_horizontal_flip:
                if image is not None:
                    imgs.append(image)

                imgs, boxes = cv2_transform.horizontal_flip_list(
                    0.5, imgs, order="HWC", boxes=boxes
                )

                if image is not None:
                    image = imgs.pop()

        elif self._split == "val" or self.cfg.DATA.JUST_CENTER:
            imgs = [cv2_transform.scale(self._crop_size[0], img) for img in imgs]
            boxes = [cv2_transform.scale_boxes(self._crop_size[0], boxes[0], height, width)]
            imgs, boxes, _ = cv2_transform.spatial_shift_crop_list(
                self._crop_size, imgs, 1, boxes=boxes, image=None
            )

            ori_aspect_ratio = width / height
            crop_aspect_ratio = self.cfg.DATA.VAL_CROP_SIZE_LARGE / self.cfg.DATA.VAL_CROP_SIZE
            assert image is None or ori_aspect_ratio - crop_aspect_ratio < self.aspect_ratio_th, (
                "Test aspect ratio difference is too large for inference with RPN"
            )

            if not self.cfg.DATA.JUST_CENTER and self._test_force_flip:
                if image is not None:
                    imgs.append(image)

                imgs, boxes = cv2_transform.horizontal_flip_list(1, imgs, order="HWC", boxes=boxes)

                if image is not None:
                    image = imgs.pop()
        else:
            raise NotImplementedError(f"Unsupported split mode {self._split}")

        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]
        imgs = [img / 255.0 for img in imgs]

        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs, img_brightness=0.4, img_contrast=0.4, img_saturation=0.4
                )

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        imgs = np.concatenate([np.expand_dims(img, axis=1) for img in imgs], axis=1)

        if not self._use_bgr:
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        boxes = cv2_transform.clip_boxes_to_image(boxes[0], imgs[0].shape[1], imgs[0].shape[2])
        if image is not None:
            image = cv2_transform.BGR2RGB(image)
            image = cv2_transform.HWC2CHW(image)
            image = torch.tensor(image)
        return imgs, boxes, image

    def __getitem__(self, idx):
        clip = self.filtered_dfs.iloc[idx] if self.filtered_dfs is not None else self.dfs.iloc[idx]

        df = self.dfs[self.dfs["patient_id"] == clip["patient_id"]]

        seq = utils.get_sequence(
            df["frame_id"].to_list().index(clip["frame_id"]),
            self._seq_len,
            self._sample_rate,
            num_frames=len(df),
            mode=self._seq_mode,
        )

        # Load all clip frames
        image_paths = df.iloc[seq]["frame_path"].to_list()

        imgs = utils.retry_load_images(
            [os.path.join(self.video_root, path) for path in image_paths],
            backend=self.cfg.ENDOVIS_DATASET.IMG_PROC_BACKEND,
        )
        if isinstance(imgs, list):
            imgs = torch.as_tensor(np.stack(imgs))

        # if self._crop_size[0] not in imgs.shape:
        #     # Convert loaded tensor into list of arrays for cv2_transform compatibility
        #     if isinstance(imgs, torch.Tensor):
        #         imgs = [img.numpy() if isinstance(img, torch.Tensor) else img for img in imgs]
        #     imgs, _, _ = self._images_and_boxes_preprocessing_cv2(imgs, boxes=None, image=None)
        # else:
        imgs = imgs.permute(3, 0, 1, 2)  # convert to CTHW
        # imgs = utils.pack_pathway_output(self.cfg, imgs)

        # convert label to numeric if needed
        all_labels = {}
        extra_data = {}

        for task in self._frame_tasks:
            if task == "steps":
                all_labels[task] = clip.get("event_id", -1)
                extra_data[f"{task}_name"] = clip.get("event_name", "unknown")
            elif task == "phases":
                all_labels[task] = clip.get("phase_id", -1)
                extra_data[f"{task}_name"] = clip.get("phase_name", "unknown")
            else:
                all_labels[task] = clip.get(f"{task}_id", -1)
                extra_data[f"{task}_name"] = clip.get(f"{task}_name", "unknown")

        frame_identifier = clip["frame_path"]

        return [imgs], all_labels, extra_data, frame_identifier

    def keyframe_mapping(self, video_idx, sec_idx, sec):
        return round(sec / 60)
        try:
            video_name = self._video_idx_to_name[video_idx]
            if video_name in self.fps_videos:
                return sec
            if video_name == "CASE014":
                complete_name = f"{video_name}/{str(sec).zfill(self.zero_fill)}.{self.image_type}"
                complete_path = os.path.join(self.cfg.ENDOVIS_DATASET.FRAME_DIR, complete_name)
                return self._image_paths[video_idx].index(complete_path)
            return round((sec * 30) / 45)
        except:
            breakpoint()

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
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/val)")
    parser.add_argument("--train_lists", type=str, default="", help="Train lists (comma-separated)")
    parser.add_argument(
        "--val_lists", type=str, default="", help="Validation lists (comma-separated)"
    )
    parser.add_argument(
        "--opts", nargs=argparse.REMAINDER, default=[], help="Override config options"
    )
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(args.cfg)

    # Override train/val lists if provided on command line
    if args.train_lists:
        cfg.ENDOVIS_DATASET.TRAIN_LISTS = args.train_lists.split(",")
    if args.test_lists:
        cfg.ENDOVIS_DATASET.TEST_LISTS = args.test_lists.split(",")

    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg = assert_and_infer_cfg(cfg)
    cfg.freeze()

    print(f"\n{'=' * 60}")
    print(f"Loading Orsi dataset for split: {args.split}")
    print(f"{'=' * 60}\n")

    dataset = Orsi(cfg, split=args.split, load=True)

    print(f"\nTotal samples: {len(dataset)}")
    print(f"Patients: {len(dataset._list_patient_ids())}")

    task_counts = {}
    for task in cfg.TASKS.TASKS:
        df = dataset.filtered_dfs if dataset.filtered_dfs is not None else dataset.dfs
        if task == "steps":
            col = "event_name"
        elif task == "phases":
            col = "phase_name"
        else:
            col = f"{task}_name"

        if col in df.columns:
            counts = df[col].value_counts().sort_index()
            task_counts[task] = counts
            print(f"\n--- {task.upper()} distribution ({len(counts)} classes) ---")
            for name, count in counts.items():
                print(f"  {name}: {count}")

    if task_counts:
        print(f"\n{'=' * 60}")
        print("Done!")
