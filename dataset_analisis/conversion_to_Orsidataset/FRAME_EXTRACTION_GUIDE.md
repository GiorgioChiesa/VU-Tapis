# Frame Extraction & ORSI to GraSP Conversion Guide

## 🎬 Overview

The updated `orsi2coco.py` converter now includes:
- **Automatic frame extraction** from MP4 videos using ffmpeg
- **Frame naming compatibility** with GraSP format (%09d.jpg)
- **Integrated conversion workflow** combining frame extraction + annotation conversion
- **Dimension auto-detection** from extracted frames

---

## ✨ Features

### 1. **Automatic Frame Extraction**
```bash
--extract-frames        # Flag to enable frame extraction
--video-root VIDEO_ROOT # Directory containing MP4 videos
--frames-output FRAMES_OUTPUT # Where to save extracted frames
```

### 2. **Frame Naming Convention**
Frames are saved as: `{frames_output}/{video_name}/%09d.jpg`

Examples:
```
frames/RARP01/000000000.jpg    # Frame 0
frames/RARP01/000150255.jpg    # Frame 150255 (last frame @ 6010.23s, 25fps)
```

This matches the GraSP frame structure exactly.

### 3. **Batch Processing**
Extract frames for multiple videos in one command:
```bash
python orsi2coco.py \
  --input annotations/ \
  --output output/coco_json/ \
  --video-root video-folder/ \
  --frames-output /path/to/frames/ \
  --extract-frames \
  --batch \
  --csv
```

---

## 🚀 Usage Examples

### Example 1: Extract Frames + Convert Annotations (Single File)

```bash
python orsi2coco.py \
  --input annotations/RARP01.json \
  --output output.json \
  --extract-frames \
  --video-root ./mp4_videos \
  --frames-output /scratch/Video_Understanding/GraSP/TAPIS/data/GraSP/frames \
  --fps 25 \
  --csv
```

**Output:**
```
output.json                                    # COCO annotations
output.json.csv                                # Frame list
/scratch/Video_Understanding/GraSP/TAPIS/data/GraSP/frames/RARP01/
  ├── 000000000.jpg
  ├── 000000001.jpg
  ├── 000000002.jpg
  └── ...
```

### Example 2: Batch Conversion with Frame Extraction

```bash
python orsi2coco.py \
  --input /path/to/annotations/ \
  --output /path/to/output_coco/ \
  --extract-frames \
  --video-root /path/to/mp4_videos/ \
  --frames-output /scratch/Video_Understanding/GraSP/TAPIS/data/GraSP/frames \
  --batch \
  --csv \
  --csv-output /path/to/output_coco/train.csv
```

### Example 3: Use Existing Frames (No Extraction)

If frames are already extracted:

```bash
python orsi2coco.py \
  --input annotations/RARP01.json \
  --output output.json \
  --frame-root /scratch/Video_Understanding/GraSP/TAPIS/data/GraSP/frames \
  --fps 25
```

The converter will:
- Read actual image dimensions from extracted frames
- Create COCO annotations with correct frame metadata
- Generate CSV files for training

### Example 4: Convert Annotations Without Frames

If you want to generate annotations first:

```bash
python orsi2coco.py \
  --input annotations/RARP01.json \
  --output output.json \
  --fps 25 \
  --csv
```

Extract frames later when ready.

---

## 🔍 How It Works

### Frame Extraction Process

1. **FFmpeg Command**
   ```bash
   ffmpeg -i video.mp4 -vf fps=25 -q:v 2 frames/RARP01/%09d.jpg
   ```
   - Extracts frames at specified FPS (default: 25)
   - Quality level 2 (high quality)
   - Saves with 9-digit zero-padded numbering

2. **Conversion Process**
   - Checks if frames already exist (skips if found)
   - Extracts frames to specified output directory
   - Automatically detects frame dimensions from extracted files
   - Creates COCO annotations with frame references

3. **Result**
   - Frame paths in COCO JSON: `RARP01/000000001.jpg`, `RARP01/000000002.jpg`, etc.
   - Actual dimensions read from extracted frames
   - CSV file with frame list for GraSP training

---

## 📊 Command-Line Arguments

### Frame Extraction Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--extract-frames` | flag | False | Enable frame extraction from videos |
| `--video-root` | str | Required | Root directory containing MP4 video files |
| `--frames-output` | str | Auto-set | Directory where extracted frames are saved |
| `--fps` | int | 25 | Video frame rate for extraction |

### Annotation Conversion Arguments

| Argument | Type | Default | Description |
| `--input` | str | Required | JSON annotation file or directory |
| `--output` | str | Required | Output COCO JSON file or directory |
| `--frame-root` | str | Optional | Root directory for reading frame dimensions |
| `--csv` | flag | False | Generate CSV frame lists |
| `--batch` | flag | False | Process multiple files |

---

## ⚠️ Important Notes

### Frame Extraction Time

Extraction speed depends on:
- Video length and resolution
- CPU performance
- Disk I/O speed

**Estimate:** 1 hour video (~2.5GB) takes ~20-30 minutes on typical hardware

### FFmpeg Requirements

The converter requires `ffmpeg` to be installed and available in PATH:

```bash
# Check FFmpeg availability
which ffmpeg
ffmpeg -version

# Install if needed:
# Ubuntu/Debian: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
# Conda: conda install -c conda-forge ffmpeg
```

### Frame Naming Convention

Frames must follow this exact format:
```
{frames_root}/{video_name}/{frame_num:09d}.jpg
```

Examples:
```
✅ Correct:
   frames/RARP01/000000000.jpg
   frames/RARP01/000150255.jpg
   
❌ Incorrect:
   frames/RARP01/0.jpg
   frames/RARP01/frame_000000000.jpg
   RARP01/frames/000000000.jpg
```

### Video File Naming

Video files must match annotation file names:
```
Annotation: RARP01.json  →  Video: RARP01.mp4
Annotation: RARP02.json  →  Video: RARP02.mp4
```

---

## 🔧 Troubleshooting

### FFmpeg Not Found

```
Error: ffmpeg not found or not installed
```

**Solution:** Install ffmpeg
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS with Homebrew
brew install ffmpeg

# Using conda
conda install -c conda-forge ffmpeg
```

### Frame Extraction Timeout

```
TimeoutError: FFmpeg extraction timeout (>1 hour)
```

**Solutions:**
1. Extract frames manually with ffmpeg
2. Use a more powerful machine
3. Extract at lower FPS if acceptable for your use case

### Video File Not Found

```
FileNotFoundError: Video file not found
```

**Solutions:**
- Verify video file exists at: `{video_root}/{video_name}.mp4`
- Check file permissions
- Verify video file is not corrupted

### Dimension Detection Issues

```
PIL.UnidentifiedImageError: cannot identify image file
```

**Solutions:**
- Check extracted frames are valid JPG files
- Verify ffmpeg extraction completed successfully
- Manually specify dimensions with `--width` and `--height`

---

## 📈 Complete Workflow Example

### Full Pipeline for RARP Dataset

```bash
#!/bin/bash

ANNOTATION_DIR="/scratch/Video_Understanding/GraSP/TAPIS/dataset_analisis/conversion_to_Orsidataset/annotations"
VIDEO_DIR="/scratch/Video_Understanding/GraSP/TAPIS/dataset_analisis/conversion_to_Orsidataset/mp4_videos"
OUTPUT_DIR="/scratch/Video_Understanding/GraSP/TAPIS/data/GraSP/annotations/orsi_coco"
FRAMES_DIR="/scratch/Video_Understanding/GraSP/TAPIS/data/GraSP/frames"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "🚀 Starting ORSI to GraSP Conversion Pipeline"
echo "=============================================="
echo "Annotations: $ANNOTATION_DIR"
echo "Videos: $VIDEO_DIR"
echo "Output COCO JSON: $OUTPUT_DIR"
echo "Frames output: $FRAMES_DIR"
echo ""

# Run conversion with frame extraction
python orsi2coco.py \
  --input "$ANNOTATION_DIR" \
  --output "$OUTPUT_DIR" \
  --extract-frames \
  --video-root "$VIDEO_DIR" \
  --frames-output "$FRAMES_DIR" \
  --fps 25 \
  --batch \
  --csv \
  --csv-output "$OUTPUT_DIR/train.csv"

echo ""
echo "✅ Conversion Complete!"
echo "├─ COCO JSON files: $OUTPUT_DIR/"
echo "├─ Extracted frames: $FRAMES_DIR/"
echo "└─ Frame list CSV: $OUTPUT_DIR/train.csv"
```

---

## 📋 Output Structure

After successful conversion:

```
GraSP/
├── annotations/
│   ├── RARP01_coco.json      # COCO format
│   ├── RARP02_coco.json
│   ├── train.csv             # Combined frame list
│   └── ...
│
├── frames/
│   ├── RARP01/
│   │   ├── 000000000.jpg
│   │   ├── 000000001.jpg
│   │   └── ...
│   │
│   ├── RARP02/
│   │   ├── 000000000.jpg
│   │   └── ...
│   │
│   └── ...
```

---

## 🎓 Integration with GraSP Training

After frame extraction and conversion, use with GraSP:

```bash
cd /scratch/Video_Understanding/GraSP/TAPIS

# Update dataset config
cat > configs/custom_orsi.yaml <<EOF
DATASETS:
  TRAIN: ('orsi_long-term_train',)
  TEST: ('orsi_long-term_test',)
  VAL: ('orsi_long-term_val',)

DATA:
  ANNOTATION_DIR: "data/GraSP/annotations/orsi_coco"
  FRAME_DIR: "data/GraSP/frames"
EOF

# Train model
python -m tapis.train --config-file configs/custom_orsi.yaml \
    MODEL.WEIGHTS data/GraSP/pretrained_models/train/LONG.pyth
```

---

## 📚 References

- FFmpeg Documentation: https://ffmpeg.org/documentation.html
- GraSP Dataset: https://cinfonia.uniandes.edu.co
- COCO Format: https://cocodataset.org/format/

---

**Version:** 2.1 (With Frame Extraction)  
**Last Updated:** February 18, 2026  
**Status:** ✅ Tested and Production-Ready
