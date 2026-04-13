#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
import os
import logging
import sys
import numpy as np

from copy import deepcopy
import pandas as pd
from tapis.datasets import cv2_transform
import torch

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
        self._video_length = max(cfg.DATA.NUM_FRAMES, cfg.MODEL.MEMORY_BANK_SIZE) if split == "train" else cfg.DATA.NUM_FRAMES
        self._seq_mode = cfg.DATA.SEQ_MODE
        self._seq_len = self._video_length * self._sample_rate
        self._num_classes = {key: n_class for key, n_class in zip(cfg.TASKS.TASKS, cfg.TASKS.NUM_CLASSES)}
        self._region_tasks = {task for task in cfg.TASKS.TASKS if task in cfg.ENDOVIS_DATASET.REGION_TASKS}
        self._frame_tasks = {task for task in cfg.TASKS.TASKS if task not in cfg.ENDOVIS_DATASET.REGION_TASKS}

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
            self._crop_size = (cfg.DATA.TEST_CROP_SIZE, cfg.DATA.TEST_CROP_SIZE_LARGE)
            self._test_force_flip = cfg.ENDOVIS_DATASET.TEST_FORCE_FLIP
            self.aspect_ratio_th = cfg.ENDOVIS_DATASET.ASPECT_RATION_TH

        #paths
        self.video_root = getattr(cfg.ENDOVIS_DATASET, "ORSI_ROOT_DIR", cfg.ENDOVIS_DATASET.FRAME_DIR)
        self.label_dir = getattr(cfg.ENDOVIS_DATASET, "ORSI_LABEL_DIR", cfg.ENDOVIS_DATASET.ANNOTATION_DIR)
        self.frame_folder = getattr(cfg.ENDOVIS_DATASET, "ORSI_FRAME_FOLDER", "Video_1fps")
        self.frame_folder_alternatives = [self.frame_folder, "IMAGES"]
        self.label_folder = getattr(cfg.ENDOVIS_DATASET, "ORSI_LABEL_FOLDER", "Label")
        self.exclude_event_names = {name.strip().lower() for name in getattr(cfg.ENDOVIS_DATASET, "EXCLUDE_EVENT_NAMES", [])}
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
    
    def generate_weight_vector(self):
        map_task={"steps": "event", "phases": "phase"}
        for task, weight_loss_by_class in zip(self.cfg.TASKS.TASKS, self.cfg.TASKS.WEIGHT_LOSS_BY_CLASS):
            if not weight_loss_by_class:
                continue
            if isinstance(weight_loss_by_class, str) and (os.path.isfile(weight_loss_by_class) or os.path.isfile(os.path.join(self.cfg.OUTPUT_DIR, "distributions", weight_loss_by_class))):
                weight_loss_by_class = os.path.abspath(os.path.join(self.cfg.OUTPUT_DIR, "distributions", weight_loss_by_class))
                
                if "csv" in weight_loss_by_class:
                    clip = pd.read_csv(weight_loss_by_class)
                elif "json" in weight_loss_by_class:
                    clip = pd.read_json(weight_loss_by_class)
                elif "xlsx" in weight_loss_by_class:
                    clip = pd.read_excel(weight_loss_by_class)
                else:
                    print(f"Unsupported file format for weight_loss_by_class: {weight_loss_by_class}. Supported formats are .csv, .json, and .xlsx")
                
                if "total_count" in list(clip.columns):
                    continue
                else:
                    print(f"Column 'total_count' not found in {weight_loss_by_class}. Please make sure the file has a column named 'total_count' with the count of samples for each class.")
                    print(f"Generating weight vector from dataset distribution instead.")
                
            clip = self.filtered_dfs.copy() if self.filtered_dfs is not None else self.dfs.copy()

            his =[]
            for id in range(self.cfg.TASKS.NUM_CLASSES[self.cfg.TASKS.TASKS.index(task)]):
                his.append({
                    "id": id,
                    "name": self.event_idx2name.get(id, "unknown") if task == "steps" else self.phase_idx2name.get(id, "unknown"),
                    "total_count": len(clip[clip[f"{map_task[task]}_id"] == id ]),

                })
            
            df = pd.DataFrame(his)
            assert df["total_count"].sum() == len(clip), f"Total count in distribution ({df['total_count'].sum()}) does not match total samples in dataset ({len(clip)})"
            if isinstance(weight_loss_by_class, str) and os.path.isabs(weight_loss_by_class):
                df.to_csv(weight_loss_by_class, index=False)
            else:
                csv_path = os.path.join(self.cfg.OUTPUT_DIR, "distributions", weight_loss_by_class)
                print(f"Weight loss by class for task {task}:\n{df}")
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                df.to_csv(csv_path, index=False)

    def _list_patient_ids(self):
        if self._split == "train":
            list_files = self.cfg.ENDOVIS_DATASET.TRAIN_LISTS
        elif self._split == "val":
            list_files = self.cfg.ENDOVIS_DATASET.TEST_LISTS
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
                if patient.endswith("_all_label"):
                    patient = patient[: -len("_all_label")]
                patients.append(patient)
            else:
                patients.append(base)
        return patients

    def _locate_label_file(self, patient):
        candidates = []
        if self.label_dir:
            candidates += [
                os.path.join(self.video_root, patient, self.label_folder, f"{patient}_all_labels.csv"),
            ]
        if self.video_root:
            candidates += [
                os.path.join(self.video_root, patient, self.label_folder, f"{patient}_all_labels.csv"),
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
        if self.exclude_event_names: #TODO: pensare se filtrare solo nel train !!!
            self.filtered_dfs = self.dfs[~self.dfs["event_name"].str.strip().str.lower().isin(self.exclude_event_names)].reset_index(drop=True)
        saving = self.filtered_dfs if self.filtered_dfs is not None else self.dfs
        saving.to_csv(os.path.join(self.cfg.OUTPUT_DIR, f"{self._split}_data.csv"), index=False)
        return   

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

        boxes = [boxes.astype('float')]

        if self._split == "train" and not self.cfg.DATA.JUST_CENTER:
            imgs, boxes = cv2_transform.random_short_side_scale_jitter_list(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
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
            boxes = [
                cv2_transform.scale_boxes(
                    self._crop_size[0], boxes[0], height, width
                )
            ]
            imgs, boxes, _ = cv2_transform.spatial_shift_crop_list(
                self._crop_size, imgs, 1, boxes=boxes, image=None
            )

            ori_aspect_ratio = (width / height)
            crop_aspect_ratio = (self.cfg.DATA.TEST_CROP_SIZE_LARGE / self.cfg.DATA.TEST_CROP_SIZE)
            assert (
                image is None
                or ori_aspect_ratio - crop_aspect_ratio < self.aspect_ratio_th
            ), f"Test aspect ratio difference is too large for inference with RPN"

            if not self.cfg.DATA.JUST_CENTER and self._test_force_flip:
                if image is not None:
                    imgs.append(image)

                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )

                if image is not None:
                    image = imgs.pop()
        else:
            raise NotImplementedError(
                "Unsupported split mode {}".format(self._split)
            )

        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]
        imgs = [img / 255.0 for img in imgs]

        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
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
        boxes = cv2_transform.clip_boxes_to_image(
            boxes[0], imgs[0].shape[1], imgs[0].shape[2]
        )
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
        
        imgs = utils.retry_load_images([os.path.join(self.video_root, path) for path in image_paths],
                                       backend=self.cfg.ENDOVIS_DATASET.IMG_PROC_BACKEND)
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
                all_labels[task] = clip.get(f"event_id", -1)
                extra_data[f"{task}_name"] = clip.get(f"event_name", "unknown")
            elif task == "phases":
                all_labels[task] = clip.get(f"phase_id", -1)
                extra_data[f"{task}_name"] = clip.get(f"phase_name", "unknown")
            else:
                all_labels[task] = clip.get(f"{task}_id", -1)
                extra_data[f"{task}_name"] = clip.get(f"{task}_name", "unknown")

        frame_identifier = clip["frame_path"]

        return [imgs], all_labels, extra_data, frame_identifier



    def keyframe_mapping(self, video_idx, sec_idx, sec):
        return round(sec/60)
        try:
            video_name = self._video_idx_to_name[video_idx]
            if video_name in self.fps_videos:
                return sec
            elif video_name=='CASE014':
                complete_name = '{}/{}.{}'.format(video_name, str(sec).zfill(self.zero_fill), self.image_type)
                complete_path = os.path.join(self.cfg.ENDOVIS_DATASET.FRAME_DIR,complete_name)
                return self._image_paths[video_idx].index(complete_path)
            else:
                return round((sec*30)/45) 
        except:
            breakpoint()
    
    def frame_name_spliting(self, video_name, sec):
        video_num = int(video_name.replace('RARP',''))
        return [video_num,sec]
    
    def frame_num_joining(self, video_num, sec):
        return f'RARP{video_num:03d}/{sec:0{self.zero_fill}d}.{self.image_type}'
    
    def frame_name_joining(self, video_name, sec):
        return f"{video_name}/IMAGES/{sec:0{self.zero_fill}d}.{self.image_type}"
        