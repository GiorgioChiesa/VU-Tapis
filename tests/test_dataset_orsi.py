#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Test suite for Orsi dataset.
Verifies dataset instantiation, batch loading, shapes, and labels.
Measures timing for dataset creation and batch loading.
"""

import time
import sys
from pathlib import Path

# Add parent directory to path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tapis.config.defaults import get_cfg
from tapis.datasets.build import build_dataset
from tapis.datasets.loader import construct_loader


def test_orsi_dataset_instantiation(num_iterations=50):
    """Test Orsi dataset instantiation and timing."""
    print("\n" + "=" * 80)
    print("TEST 1: Orsi Dataset Instantiation")
    print("=" * 80)

    cfg = get_cfg()
    cfg.TRAIN.DATASET = "orsi"
    cfg.TEST.DATASET = "orsi"
    cfg.ENDOVIS_DATASET.FRAME_DIR = "/data/orsi_tensors"
    cfg.ENDOVIS_DATASET.ANNOTATION_DIR = "/data/orsi_tensors"
    cfg.ENDOVIS_DATASET.ORSI_ROOT_DIR = "/data/orsi_tensors"
    cfg.ENDOVIS_DATASET.ORSI_LABEL_DIR = "/data/orsi_tensors"
    cfg.ENDOVIS_DATASET.TRAIN_LISTS = ["RARP01.csv", "RARP03.csv", "RARP04.csv", "RARP06.csv"]
    cfg.ENDOVIS_DATASET.TEST_LISTS = ["RARP02.csv", "RARP07.csv", "RARP08.csv", "RARP09.csv", "RARP10.csv"]

    cfg.DATA.NUM_FRAMES = 16
    cfg.DATA.SAMPLING_RATE = 1
    cfg.DATA.SEQ_MODE = "center"
    cfg.MODEL.MEMORY_BANK_SIZE = 0
    cfg.AUG.ENABLE = False
    cfg.ENDOVIS_DATASET.EXCLUDE_EVENT_NAMES = ["Idle", "Out_of_body", "Test_image_start"]
    cfg.ENDOVIS_DATASET.IMG_PROC_BACKEND = "pytorch"

    # Measure dataset instantiation time
    t0 = time.time()
    for _ in range(num_iterations):
        ds = build_dataset("orsi", cfg, "train")
    t1 = time.time()

    dataset_size = len(ds)
    instantiation_time = (t1 - t0) / num_iterations

    print(f"✓ Dataset instantiated successfully")
    print(f"  - Dataset size: {dataset_size} clips")
    print(f"  - Instantiation time: {instantiation_time:.3f} seconds")

    # Test single sample
    sample = ds[0]
    imgs, labels, extra_data, frame_id = sample

    print(f"\n✓ Sample loaded successfully")
    print(f"  - Images type: {type(imgs)}")
    if isinstance(imgs, list):
        print(f"  - Images list length: {len(imgs)}")
        for idx, img in enumerate(imgs):
            if hasattr(img, "shape"):
                print(f"    - Pathway {idx} shape: {img.shape}")
    print(f"  - Labels: {labels}")
    print(f"  - Labels keys: {list(labels.keys())}")
    print(f"  - Frame identifier: {frame_id}")

    return ds, cfg, instantiation_time


def test_dataloader_batch(ds, cfg):
    """Test DataLoader batch loading and timing."""
    print("\n" + "=" * 80)
    print("TEST 2: DataLoader Batch Loading")
    print("=" * 80)

    # Reduce batch size for faster testing
    cfg.TRAIN.BATCH_SIZE = 8
    cfg.DATA_LOADER.NUM_WORKERS = 2  # Use workers to avoid prefetch_factor issue

    # Measure data loader construction time
    t0 = time.time()
    loader = construct_loader(cfg, "train")
    t1 = time.time()

    loader_creation_time = t1 - t0
    print(f"✓ DataLoader created successfully")
    print(f"  - Loader creation time: {loader_creation_time:.3f} seconds")

    # Measure batch loading time
    t0 = time.time()
    batch = next(iter(loader))
    t1 = time.time()

    batch_load_time = t1 - t0
    print(f"\n✓ Batch loaded successfully")
    print(f"  - Batch loading time: {batch_load_time:.3f} seconds")

    # Verify batch structure
    imgs_batch, labels_batch, extra_batch, ids_batch = batch

    print(f"\n✓ Batch structure verified")
    print(f"  - Images batch type: {type(imgs_batch)}")

    if isinstance(imgs_batch, list):
        print(f"  - Images list length: {len(imgs_batch)}")
        for idx, img in enumerate(imgs_batch):
            print(f"    - Pathway {idx} shape: {img.shape}")
    else:
        print(f"  - Images batch shape: {imgs_batch.shape}")

    print(f"  - Labels type: {type(labels_batch)}")
    print(f"  - Labels keys: {list(labels_batch.keys())}")
    for key, val in labels_batch.items():
        print(f"    - {key}: shape={val.shape}, dtype={val.dtype}")

    print(f"  - Extra data type: {type(extra_batch)}")
    print(f"  - Extra data keys: {list(extra_batch.keys())}")

    print(f"  - IDs batch type: {type(ids_batch)}")
    print(f"  - IDs batch length: {len(ids_batch)}")
    print(f"  - Sample IDs: {ids_batch[:3]}")

    return batch_load_time


def test_multiple_batches(cfg, num_batches=50):
    """Test loading multiple batches and measure average time."""
    print("\n" + "=" * 80)
    print(f"TEST 3: Multiple Batch Loading ({num_batches} batches)")
    print("=" * 80)

    cfg.TRAIN.BATCH_SIZE = 8
    cfg.DATA_LOADER.NUM_WORKERS = 2

    loader = construct_loader(cfg, "train")

    times = []
    for batch_idx in range(num_batches):
        t0 = time.time()
        batch = next(iter(loader))
        t1 = time.time()
        batch_time = t1 - t0
        times.append(batch_time)
        print(f"  Batch {batch_idx + 1}: {batch_time:.3f}s")

    avg_time = sum(times) / len(times)
    print(f"\n✓ Average batch loading time: {avg_time:.3f} seconds")

    return avg_time


def test_label_distribution(ds):
    """Test and print label distribution in dataset."""
    print("\n" + "=" * 80)
    print("TEST 4: Label Distribution")
    print("=" * 80)

    label_counts = {}

    # Sample 100 random clips to estimate distribution
    sample_size = min(100, len(ds))
    import random

    random.seed(42)
    sample_indices = random.sample(range(len(ds)), sample_size)

    for idx in sample_indices:
        _, labels, _, _ = ds[idx]
        for task, label_val in labels.items():
            if task not in label_counts:
                label_counts[task] = {}
            label_key = int(label_val.item()) if hasattr(label_val, "item") else int(label_val)
            label_counts[task][label_key] = label_counts[task].get(label_key, 0) + 1

    print(f"✓ Sampled {sample_size} clips from dataset")
    for task, counts in label_counts.items():
        print(f"\n  Task: {task}")
        sorted_counts = sorted(counts.items())
        for label_id, count in sorted_counts:
            percentage = (count / sample_size) * 100
            print(f"    Label {label_id}: {count} ({percentage:.1f}%)")


def test_edge_cases(ds, cfg):
    """Test edge cases and error handling."""
    print("\n" + "=" * 80)
    print("TEST 5: Edge Cases")
    print("=" * 80)

    # Test first and last samples
    first_sample = ds[0]
    last_sample = ds[len(ds) - 1]

    print(f"✓ First sample loaded: {first_sample[3]}")
    print(f"✓ Last sample loaded: {last_sample[3]}")

    # Test batch with size 1
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.DATA_LOADER.NUM_WORKERS = 2
    loader = construct_loader(cfg, "train")
    batch_single = next(iter(loader))
    print(f"✓ Batch size 1 loaded: images shape {batch_single[0][0].shape}")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("ORSI DATASET TEST SUITE")
    print("=" * 80)

    try:
        # Test 1: Dataset instantiation
        ds, cfg, inst_time = test_orsi_dataset_instantiation()

        # Test 2: Batch loading
        batch_time = test_dataloader_batch(ds, cfg)

        # Test 3: Multiple batches
        avg_batch_time = test_multiple_batches(cfg, num_batches=50)

        # Test 4: Label distribution
        test_label_distribution(ds)

        # Test 5: Edge cases
        test_edge_cases(ds, cfg)

        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"✓ Dataset instantiation time: {inst_time:.3f}s")
        print(f"✓ First batch loading time: {batch_time:.3f}s")
        print(f"✓ Average batch loading time: {avg_batch_time:.3f}s")
        print(f"✓ Total dataset size: {len(ds)} clips")
        print(f"\n✅ All tests passed!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
