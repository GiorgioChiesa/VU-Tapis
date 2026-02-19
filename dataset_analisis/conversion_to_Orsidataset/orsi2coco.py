import json
import os
import csv
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import argparse
import cv2
from tqdm import tqdm

# FPS configuration - adjust based on your video fps
FPS = 60  # Change this to match your video frame rate

# Try to import PIL for reading image dimensions
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ===== RARP Event Timing Specifications from "Phases of Robot-Assisted Radical Prostatectomy" =====
# Events categorized by temporal precision requirement

# Events that occur in 1 frame (exact frame)
INSTANT_FRAME_EVENTS = {
    "Out of body", "Instrument swap: removal", "Instrument swap: insertion",
    "Insert gauze", "Remove gauze", "Insert hemostatic agens", "Remove hemostatic agens",
    "Test image start", "Test image stop", "Inside abdomen", "Instrument insertion",
    "Adhesion removal", "Fat removal", "Remove needle bladder stretch stitch",
    "Needle removal DVC ligation", "V-lock", "Cutting the needles", "Removing the needles",
    "Threads removal", "Vessel loop removal", "Hemolock clip removal", "Endobag removal",
    "Drain placement", "Removal of robotic instruments", "Camera out of body", "Camera stop"
}

# Events with ±1 second margin
MARGIN_1SEC_EVENTS = {
    "Unsuccesful clip placement", "Hemostatic metal clip placement",
    "Incision peritoneum - left", "Incision peritoneum - right",
    "Incision of the fascia - left", "Incision of the fascia - right",
    "Placement stitch for bladder stretch", "Start dissection",
    "Visualisation of urethra opening", "Grasping catheter tip",
    "Continue posterior dissection", "Hemolock clip on bladder pedicle attached to prostate",
    "Identify and dissect vas deferens - left", "Clip or coagulate vas deferens - left",
    "Identification and clipping of SV arteries - left",
    "Identify and dissect vas deferens - right", "Clip or coagulate vas deferens - right",
    "Identification and clipping of SV arteries - right",
    "Lift both seminal vesicles", "Incision of Denonvilliers fascia",
    "Lift right seminal vesicle", "Start dissection and cutting right pedicle",
    "Hemolock clip on right pedicle", "Metal clip on right pedicle",
    "Lift left seminal vesicle", "Start dissection and cutting left pedicle",
    "Hemolock clip on left pedicle", "Metal clip on left pedicle",
    "Start dissection DVC", "Stitch in DVC before apical dissection",
    "Transection of the urethra", "Tighten endobag",
    "Stitch in DVC after apical dissection", "Stitch of posterior reconstruction",
    "Stitch in bladder", "Stitch in urethra", "Tie suture",
    "Final reinforcing suture", "Endobag removal"
}

# Events with ±5 seconds margin
MARGIN_5SEC_EVENTS = {
    "Port placement", "Leak test"
}

# Paired events that form START/END ranges
PAIRED_EVENT_TEMPLATES = [
    ("Out of body", "Back inside body"),
    ("Insert gauze", "Remove gauze"),
    ("Insert hemostatic agens", "Remove hemostatic agens"),
]

class ORSI2COCO:
    """
    Convert ORSI dataset format (RARP01.json) to COCO format compatible with GraSP

    Source format: RARP01.json with EVENTS (steps/azioni) and PHASES (fasi chirurgiche)
    Target format: COCO JSON with phases_categories and steps_categories
    
    Semantic mapping:
    - RARP Events → COCO steps_categories (EVENTS = STEPS)
    - RARP Phases → COCO phases_categories
    
    Event classification:
    - Instant events: Single timestamp, active for margin duration
    - Paired events: START + END timestamps create continuous ranges
    """

    def __init__(self, fps: int = FPS):
        self.fps = fps
        self.image_id_counter = 1
        self.annotation_id_counter = 1
        self.phase_to_id = {}
        self.step_to_id = {}
        self.annotated_frames = set()
        self.dimensions_cache = {}
        self.event_margins = {}  # Will be populated with event timing margins
        self._init_event_margins()
    
    def reset_counters(self):
        """Reset image and annotation ID counters for a new video"""
        self.image_id_counter = 1
        self.annotation_id_counter = 1
        self.phase_to_id.clear()
        self.step_to_id.clear()
        self.annotated_frames.clear()
        self.dimensions_cache.clear()


    def _init_event_margins(self):
        """Initialize event timing margins based on RARP specifications"""
        # Each event has a temporal margin (in seconds) based on RARP document
        for event_name in INSTANT_FRAME_EVENTS:
            self.event_margins[event_name] = 0.0  # Exact frame

        for event_name in MARGIN_1SEC_EVENTS:
            self.event_margins[event_name] = 1.0  # ±1 second margin

        for event_name in MARGIN_5SEC_EVENTS:
            self.event_margins[event_name] = 5.0  # ±5 seconds margin

    def get_event_margin(self, event_name: str) -> float:
        """Get temporal margin (in seconds) for an event based on RARP specifications"""
        return self.event_margins.get(event_name, 0.0)

    def extract_frames_from_video(self, video_path: str, output_dir: str, 
                                   video_name: str, skip_existing: bool = True, 
                                   sorted_frames: list = []) -> Tuple[int, str]:
        """
        Extract frames from MP4 video using ffmpeg.
        
        Frames are saved as: {output_dir}/{video_name}/%09d.jpg
        This matches GraSP frame naming convention.
        
        Args:
            video_path: Path to input MP4 video
            output_dir: Root directory for frame output
            video_name: Name of the video (folder name)
            skip_existing: If True, skip extraction if frames already exist
            sorted_frames: Optional list of specific frame numbers to extract (if empty, extract all frames)
            
        Returns:
            Tuple of (total_frames_extracted, frame_output_dir)
        """
        frame_output_dir = os.path.join(output_dir, video_name)
        if not os.path.exists(video_path):
            print(f"⚠️  Video file not found: {video_path}. Skipping frame extraction.")
            return 0, frame_output_dir
        
        # Check if frames already exist
        if skip_existing and os.path.exists(frame_output_dir):
            existing_frames = len([f for f in os.listdir(frame_output_dir) 
                                  if f.endswith('.jpg')])
            if existing_frames > 0:
                print(f"  ℹ️  Frames already exist for {video_name}: {existing_frames} frames")
                return existing_frames, frame_output_dir
        
        # Create output directory
        os.makedirs(frame_output_dir, exist_ok=True)
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"  📹 Extracting frames from {video_name}...")
        
        try:
            
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            
            total_frames = int(len(sorted_frames) )
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  🎞️  Video FPS: {fps:.2f}, Total frames to extract: {total_frames}")
            
            extracted_frames = 0
            
            with tqdm(total=total_frames, desc=f"Extracting {video_name}", unit="frame") as pbar:
                for frame_num in sorted_frames:
                    # Set video position to the desired frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_path = os.path.join(frame_output_dir, f"{frame_num:09d}.jpg")
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    extracted_frames += 1
                    pbar.update(1)
            
            cap.release()
            print(f"  ✅ Extracted {extracted_frames} frames to {frame_output_dir} \n")
            
            return extracted_frames, frame_output_dir
            
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"FFmpeg extraction timeout (>1 hour) for {video_name}")
        except Exception as e:
            raise RuntimeError(f"Frame extraction failed: {str(e)}")

    def seconds_to_frame(self, seconds: float) -> int:
        """Convert time in seconds to frame number"""
        return int(seconds * self.fps)

    def get_image_dimensions(self, image_path: str, default_width: int = 1280,
                            default_height: int = 800) -> Tuple[int, int]:
        """
        Get actual image dimensions from file, with fallback to defaults

        Args:
            image_path: Full path to image file
            default_width: Default width if image not found
            default_height: Default height if image not found

        Returns:
            (width, height) tuple
        """
        # Check cache first
        if image_path in self.dimensions_cache:
            return self.dimensions_cache[image_path]

        # If PIL not available, use defaults
        if not PIL_AVAILABLE:
            return (default_width, default_height)

        # Try to read from file
        try:
            if os.path.exists(image_path):
                with Image.open(image_path) as img:
                    width, height = img.size
                    self.dimensions_cache[image_path] = (width, height)
                    return (width, height)
        except Exception as e:
            # Silently fallback to defaults on error
            pass

        return (default_width, default_height)

    def identify_instant_events(self, events_data: Dict) -> Set[str]:
        """
        Identify which events are instant vs paired range events.
        
        From RARP document:
        - Instant: Single timestamp events (most events)
        - Paired: START→END events like "Out of body"↔"Back inside body"
          where consecutive timestamp pairs form a continuous range
        """
        paired_events = set()
        all_event_names = set()
        
        # Collect all event names
        for category in events_data.values():
            all_event_names.update(category.keys())
        
        # Check for paired events from predefined templates
        for start_name, end_name in PAIRED_EVENT_TEMPLATES:
            if start_name in all_event_names and end_name in all_event_names:
                paired_events.add((start_name, end_name))
        
        # Generic pattern detection: "Start X" + "End X", "Insert" + "Remove", etc.
        event_list = sorted(list(all_event_names))
        for event in event_list:
            event_lower = event.lower()
            
            # Look for complementary pairs in same category first
            for category_data in events_data.values():
                if event not in category_data:
                    continue
                    
                for other_event in category_data.keys():
                    other_lower = other_event.lower()
                    
                    # Check for start/end patterns
                    if ("start" in event_lower and "end" in other_lower) or \
                       ("begin" in event_lower and "end" in other_lower) or \
                       ("insert" in event_lower and "remove" in other_lower) or \
                       ("inside" in event_lower and "out of" in other_lower):
                        
                        # Verify they have matching timestamp counts
                        try:
                            event_timestamps = category_data[event]
                            other_timestamps = category_data[other_event]
                            # If both have even number of timestamps or matching lengths, likely paired
                            if event_timestamps and other_timestamps and \
                               len(event_timestamps) == len(other_timestamps):
                                paired_events.add(set([event, other_event]))
                        except:
                            pass
        
        # All other events are instant
        p_event = set(x for sublist in paired_events for x in sublist)

        instant_events = all_event_names - p_event
        
        return instant_events, paired_events

    def load_annotation_file(self, json_path: str) -> Dict:
        """Load RARP01.json annotation file"""
        with open(json_path, 'r') as f:
            return json.load(f)

    def build_phase_categories(self, phases_data: Dict) -> List[Dict]:
        """Convert PHASES from RARP01 to COCO phases_categories format"""
        categories = [
            {
                "id": 0,
                "name": "Idle",
                "description": "Idle",
                "supercategory": "phase"
            }
        ]

        self.phase_to_id["Idle"] = 0
        phase_id = 1

        for phase_name in sorted(phases_data.keys()):
            if phase_name not in ["Idle"]:
                categories.append({
                    "id": phase_id,
                    "name": phase_name.replace(" ", "_"),
                    "description": phase_name,
                    "supercategory": "phase"
                })
                self.phase_to_id[phase_name] = phase_id
                phase_id += 1

        return categories

    def build_step_categories(self, events_data: Dict) -> List[Dict]:
        """Convert EVENTS from RARP01 to COCO steps_categories format"""
        categories = [
            {
                "id": 0,
                "name": "Idle",
                "description": "Idle",
                "supercategory": "step"
            }
        ]

        self.step_to_id["Idle"] = 0
        step_id = 1

        # Flatten all events from all categories
        seen_steps = set()

        for category_name in sorted(events_data.keys()):
            for event_name in sorted(events_data[category_name].keys()):
                if event_name not in seen_steps:
                    seen_steps.add(event_name)
                    categories.append({
                        "id": step_id,
                        "name": event_name.replace(" ", "_"),
                        "description": event_name,
                        "supercategory": "step"
                    })
                    self.step_to_id[event_name] = step_id
                    step_id += 1

        return categories

    def get_frame_phase(self, frame_num: int, phases_data: Dict) -> int:
        """Determine which phase a frame belongs to based on timing"""
        frame_time = frame_num / self.fps

        # Check each phase to find which one contains this time
        for phase_name, phase_timing in phases_data.items():
            start_times = phase_timing.get("START", [])
            end_times = phase_timing.get("END", [])

            if not start_times or not end_times:
                continue

            # Assuming there's one continuous segment per phase in the main structure
            start_time = start_times[0] if start_times else None
            end_time = end_times[0] if end_times else None

            if start_time is not None and end_time is not None:
                if start_time <= frame_time <= end_time:
                    return self.phase_to_id.get(phase_name, 0)

        return 0  # Idle if no phase matches

    def get_frame_step(self, frame_num: int, events_data: Dict, instant_events: Set[str], paired_events: Set[Tuple[str,str]]) -> int:
        """
        Determine which step/event a frame belongs to based on timing and RARP semantics.

        Priority:
        1. Paired events: Check if frame is within START→END range
        2. Instant events: Check if frame is within margin window of event timestamp
        3. Most recent event: Fallback to closest event before this frame
        
        RARP timing margins from document:
        - Instant (1 frame): exact match only
        - ±1 second margin: ±1s window around timestamp
        - ±5 seconds margin: ±5s window around timestamp
        """
        frame_time = frame_num / self.fps

        # Priority 1: Check paired (range) events
        # These are START→END pairs where consecutive timestamps form a range
        paired_events = set()
        for category, events in events_data.items():
            for start_event, end_event in paired_events:
                if start_event in category and end_event in category:
                    start_timestamps = category[start_event]
                    end_timestamps = category[end_event]
                    # Pair consecutive timestamps as start-end
                    sorted_start_ts = sorted(start_timestamps)
                    sorted_end_ts = sorted(end_timestamps)
                    for s, e in zip(sorted_start_ts, sorted_end_ts):
                        if s <= frame_time < e:
                            return self.step_to_id.get(start_event, 0)
                        if frame_time == e:
                            return self.step_to_id.get(end_event, 0)

        # Priority 2: Check instant events with RARP timing margins
        closest_instant_event = None
        closest_instant_time = -float('inf')
        
        for category_name, events in events_data.items():
            for event_name, timestamps in events.items():
                if event_name in instant_events:
                    margin = self.get_event_margin(event_name)
                    for timestamp in timestamps:
                        # Check if frame is within margin window of this event
                        if timestamp - margin <= frame_time <= timestamp + margin:
                            # If multiple instant events active, use most recent
                            if timestamp > closest_instant_time:
                                closest_instant_time = timestamp
                                closest_instant_event = event_name

        if closest_instant_event:
            return self.step_to_id.get(closest_instant_event, 0)

        # Priority 3: Fallback to most recent event before this frame
        closest_event = None
        closest_time = -float('inf')

        for category_name, events in events_data.items():
            for event_name, timestamps in events.items():
                for timestamp in timestamps:
                    if timestamp <= frame_time and timestamp > closest_time:
                        closest_time = timestamp
                        closest_event = event_name

        if closest_event:
            return self.step_to_id.get(closest_event, 0)

        return 0  # Idle if no event matches

    def create_image_entry(self, video_name: str, frame_num: int,
                          width: int = 1280, height: int = 800,
                          frame_root: Optional[str] = None) -> Dict:
        """
        Create COCO image entry

        Args:
            video_name: Name of the video
            frame_num: Frame number
            width: Default frame width (used if image cannot be read)
            height: Default frame height (used if image cannot be read)
            frame_root: Optional root directory for frames to read actual dimensions

        Returns:
            COCO image entry dictionary
        """
        file_name = f"{video_name}/{frame_num:09d}.jpg"

        # Try to get actual dimensions from image if frame_root is provided
        actual_width = width
        actual_height = height

        if frame_root:
            image_path = os.path.join(frame_root, file_name)
            actual_width, actual_height = self.get_image_dimensions(
                image_path, width, height
            )

        image_entry = {
            "id": self.image_id_counter,
            "file_name": file_name,
            "width": actual_width,
            "height": actual_height,
            "date_captured": "",
            "license": 1,
            "coco_url": "",
            "flickr_url": "",
            "video_name": video_name,
            "frame_num": frame_num
        }
        self.image_id_counter += 1
        return image_entry

    def create_annotation_entry(self, image_id: int, image_name: str,
                               phase_id: int, step_id: int) -> Dict:
        """Create COCO annotation entry"""
        annotation_entry = {
            "id": self.annotation_id_counter,
            "image_id": image_id,
            "image_name": image_name,
            "phases": phase_id,
            "steps": step_id
        }
        self.annotation_id_counter += 1
        return annotation_entry

    def convert_annotation(self, json_path: str, video_name: str,
                         output_json_path: str, frame_step: int = 46,
                         width: int = 1280, height: int = 800,
                         frame_root: Optional[str] = None):
        """
        Convert a single RARP annotation file to COCO format

        Args:
            json_path: Path to RARP01.json file
            video_name: Name of the video (used in file paths)
            output_json_path: Path to output COCO JSON file
            frame_step: Extract frames every N frames (default 46 ~ 1.84s at 25fps)
            width: Default video frame width (used if images not available)
            height: Default video frame height (used if images not available)
            frame_root: Optional root directory for frames to read actual dimensions from images
        """
        # Load input annotation
        data = self.load_annotation_file(json_path)

        events_data = data.get("EVENTS", {})
        phases_data = data.get("PHASES", {})

        # Identify instant events
        instant_events, paired_events = self.identify_instant_events(events_data)

        # Build COCO structure
        coco_output = {
            "info": {
                "description": "ORSI Dataset",
                "url": "https://www.orsi-online.com/",
                "version": "1",
                "year": "2026",
                "contributor": "ORSI"
            },
            "phases_categories": self.build_phase_categories(phases_data),
            "steps_categories": self.build_step_categories(events_data),
            "images": [],
            "annotations": []
        }

        # Determine total duration from all timestamps
        max_time = 0
        
        # Find max time from events
        for category in events_data.values():
            for timestamps in category.values():
                if timestamps:
                    max_time = max(max_time, max(timestamps))

        # Find max time from phases
        for category in phases_data.values():
            end_times = category.get("END", [])
            if end_times:
                max_time = max(max_time, max(end_times))

        total_frames = self.seconds_to_frame(max_time) + 1

        # Collect frames with important annotations (events, phase boundaries)
        frames_with_annotations = set()

        # Add frames around instant events
        for category in events_data.values():
            for event_name, timestamps in category.items():
                if event_name in instant_events:
                    margin = self.get_event_margin(event_name)
                    for timestamp in timestamps:
                        # Add frames within the event's temporal margin (from RARP specs)
                        start_frame = max(0, self.seconds_to_frame(timestamp - margin))
                        end_frame = min(total_frames - 1, 
                                       self.seconds_to_frame(timestamp + margin))
                        for f in range(start_frame, end_frame + 1, max(1, frame_step // 2)):
                            frames_with_annotations.add(f)

        # Add frames for paired events (ranges)
        for category in events_data.values():
            for start_event, end_event in paired_events:
                if start_event in category and end_event in category:
                    start_timestamps = category[start_event]
                    end_timestamps = category[end_event]
                    # Pair consecutive timestamps as start-end
                    sorted_start_ts = sorted(start_timestamps)
                    sorted_end_ts = sorted(end_timestamps)
                    for s, e in zip(sorted_start_ts, sorted_end_ts):
                        start_frame = max(0, self.seconds_to_frame(s))
                        end_frame = min(total_frames - 1, self.seconds_to_frame(e))
                        # Add frames at boundaries and some in between
                        frames_with_annotations.add(start_frame)
                        frames_with_annotations.add(end_frame)
                        for f in range(start_frame, end_frame + 1, max(1, frame_step // 2)):
                            frames_with_annotations.add(f)
                


            # for event_name, timestamps in category.items():
            #     if event_name not in instant_events and len(timestamps) >= 2:
            #         # Pair consecutive timestamps as start-end
            #         sorted_ts = sorted(timestamps)
            #         for i in range(0, len(sorted_ts) - 1, 2):
            #             start_time = sorted_ts[i]
            #             end_time = sorted_ts[i + 1]
            #             start_frame = max(0, self.seconds_to_frame(start_time))
            #             end_frame = min(total_frames - 1, self.seconds_to_frame(end_time))
            #             # Add frames at boundaries and some in between
            #             frames_with_annotations.add(start_frame)
            #             frames_with_annotations.add(end_frame)
            #             for f in range(start_frame, end_frame + 1, max(1, frame_step // 2)):
            #                 frames_with_annotations.add(f)

        # Add frames at phase boundaries
        for phase_name, phase_timing in phases_data.items():
            start_times = phase_timing.get("START", [])
            end_times = phase_timing.get("END", [])
            if start_times:
                start_frame = max(0, self.seconds_to_frame(start_times[0]))
                frames_with_annotations.add(start_frame)
            if end_times:
                end_frame = min(total_frames - 1, self.seconds_to_frame(end_times[0]))
                frames_with_annotations.add(end_frame)

        # Generate frames: combine regular sampling with annotation frames
        all_frames = set()

        # Add regularly sampled frames
        for frame_num in range(0, total_frames, frame_step):
            all_frames.add(frame_num)

        # Add all frames with important annotations
        all_frames.update(frames_with_annotations)

        # Sort frames
        sorted_frames = sorted(list(all_frames))

        # Create images and annotations
        for frame_num in sorted_frames:
            if frame_num < 0 or frame_num >= total_frames:
                continue

            image_entry = self.create_image_entry(video_name, frame_num, width, height, frame_root)
            coco_output["images"].append(image_entry)

            # Get phase and step for this frame
            phase_id = self.get_frame_phase(frame_num, phases_data)
            step_id = self.get_frame_step(frame_num, events_data, instant_events, paired_events)

            annotation_entry = self.create_annotation_entry(
                image_entry["id"],
                image_entry["file_name"],
                phase_id,
                step_id
            )
            coco_output["annotations"].append(annotation_entry)

            # Track annotated frames
            if frame_num in frames_with_annotations:
                self.annotated_frames.add(frame_num)

        # Save output JSON
        with open(output_json_path, 'w') as f:
            json.dump(coco_output, f, indent=4)

        print(f"Converted: {json_path}")
        print(f"  Total duration: {max_time:.2f}s ({total_frames} frames)")
        print(f"  Extracted frames: {len(coco_output['images'])}")
        print(f"  Frames with key annotations: {len(frames_with_annotations)}")
        print(f"  Instant events identified: {len(instant_events)}")
        print(f"  Paired events identified: {len([e for e in events_data.values()])}")
        print(f"  Phases: {len(coco_output['phases_categories'])}")
        print(f"  Steps: {len(coco_output['steps_categories'])}")
        print(f"  Output: {output_json_path}")

        return coco_output, sorted_frames

    def generate_csv_from_coco(self, coco_data: Dict, video_name: str,
                               output_csv_path: str, partition_index: int = 1):
        """
        Generate CSV file from COCO annotations

        CSV format: VIDEO_NAME PARTITION_IDX FRAME_INDEX FILE_PATH

        Note: FILE_PATH is the exact path from the coco_data['images'][]['file_name']
        which includes the actual frame number from the video
        """
        csv_rows = []

        for idx, image in enumerate(coco_data.get("images", [])):
            row = [
                video_name,
                str(partition_index),
                str(idx),
                image["file_name"]  # Use the file_name with actual frame_num
            ]
            csv_rows.append(row)

        # Write CSV
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=' ')
            writer.writerows(csv_rows)

        print(f"  CSV file: {output_csv_path}")


    def convert_batch(self, annotation_dir: str, output_dir: str,
                     frame_step: int = 46, width: int = 1280, height: int = 800,
                     generate_csv: bool = True, csv_output_path: str = None,
                     frame_root: Optional[str] = None, video_root: Optional[str] = None):
        """
        Convert all annotation files in a directory

        Args:
            annotation_dir: Directory containing RARP*.json files
            output_dir: Directory for output COCO JSON files
            frame_step: Extract frames every N frames
            width: Default video frame width (used if images not available)
            height: Default video frame height (used if images not available)
            generate_csv: Whether to generate a combined CSV file
            csv_output_path: Path for combined CSV file (default: output_dir/train.csv)
            frame_root: Optional root directory for frames to read actual dimensions from images
            video_root: Optional root directory for videos to extract frames from
        """
        os.makedirs(output_dir, exist_ok=True)

        annotation_files = sorted(Path(annotation_dir).glob("RARP*.json"))

        print(f"Found {len(annotation_files)} annotation files")

        if csv_output_path is None:
            csv_output_path = os.path.join(output_dir, "train.csv")

        all_csv_rows = []

        for ann_idx, json_file in enumerate(annotation_files):
            # Extract video name from json filename (e.g., RARP01.json -> RARP01)
            video_name = json_file.stem  # Without .json extension

            # Reset counters for each video
            self.image_id_counter = 1
            self.annotation_id_counter = 1
            self.phase_to_id = {}
            self.step_to_id = {}
            self.annotated_frames = set()

            output_json = os.path.join(output_dir, f"{video_name}_coco.json")

            try:
                coco_data, sorted_frames = self.convert_annotation(
                    str(json_file),
                    video_name,
                    output_json,
                    frame_step,
                    width,
                    height,
                    frame_root
                )

                # Generate individual CSV for this video
                if generate_csv:
                    video_csv = os.path.join(output_dir, f"{video_name}.csv")
                    self.generate_csv_from_coco(coco_data, video_name, video_csv, partition_index=1)

                    # Add to combined CSV
                    for idx, image in enumerate(coco_data.get("images", [])):
                        row = [video_name, "1", str(idx), image["file_name"]]
                        all_csv_rows.append(row)

                if video_root and os.path.isdir(video_root):
                    video_path = os.path.join(video_root, f"{video_name}.mp4")
                    print(f"\n🎬 Frame Extraction Mode:")
                    self.extract_frames_from_video(video_path, frame_root, video_name, sorted_frames=sorted_frames)



            except Exception as e:
                print(f"  Error processing {json_file}: {str(e)}")

        # Write combined CSV
        if generate_csv and all_csv_rows:
            with open(csv_output_path, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=' ')
                writer.writerows(all_csv_rows)
            print(f"\nGenerated combined CSV: {csv_output_path}")
            print(f"Total rows in CSV: {len(all_csv_rows)}")



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Convert ORSI dataset to COCO format with optional frame extraction")
    parser.add_argument("--input", type=str, help="Input JSON file or directory with JSON files")
    parser.add_argument("--output", type=str, help="Output JSON file or directory")
    parser.add_argument("--fps", type=int, default=25, help="Video FPS (default: 25)")
    parser.add_argument("--frame-step", type=int, default=46, help="Extract every N frames (default: 46)")
    parser.add_argument("--width", type=int, default=1280, help="Video width (default: 1280)")
    parser.add_argument("--height", type=int, default=800, help="Video height (default: 800)")
    # parser.add_argument("--csv", action="store_true", help="Generate CSV files")
    parser.add_argument("--csv-output", type=str, help="Path for combined CSV file (default: output_dir/train.csv)")
    parser.add_argument("--frame-root", type=str, help="Root directory for existing frames (reads dimensions)")
    # parser.add_argument("--extract-frames", action="store_true", help="Extract frames from videos using ffmpeg")
    parser.add_argument("--video-root", type=str, help="Root directory containing MP4 videos (for --extract-frames)")
    # parser.add_argument("--frames-output", type=str, help="Directory for saved frames (default: frame-root or current/frames)")
    # parser.add_argument("--batch", action="store_true", help="Process all RARP*.json files in input directory")

    args = parser.parse_args()
    
    # Validate arguments
    if args.video_root and os.path.isdir(args.video_root):
        args.extract_frames = True  # Ensure frame extraction if video_root is provided
    else:
        args.extract_frames = False
    if args.extract_frames and (not args.video_root or not os.path.isdir(args.video_root)):
        print("❌ Error: --extract-frames requires --video-root or --video_root to be a directory")
        exit(1)
    if not args.input or not args.output:
        print("--input and --output are required")
        exit(1)
    if os.path.isdir(args.input) and not os.path.isdir(args.output):
        print("❌ Error: When --input is a directory, --output must also be a directory")
        exit(1)
    if args.csv_output:
        args.csv = True  # Ensure CSV generation if csv_output is specified
    else:
        args.csv = False
    


        
    


    converter = ORSI2COCO(fps=args.fps)
    if os.path.isdir(args.input):
        # Extract frames if requested
        frames_root = args.frame_root
        # if args.extract_frames:
        #     print(f"\n🎬 Frame Extraction Mode:")
        #     print(f"   Video root: {args.video_root}")
        #     print(f"   Frames output: {args.frame_root}")
            
        #     # Find all annotation files
        #     annotation_files = sorted(Path(args.input).glob("RARP*.json"))
        #     for json_file in annotation_files:
        #         video_name = json_file.stem
        #         video_path = os.path.join(args.video_root, f"{video_name}.mp4")
        #         print(f"\n📄 {video_name}:")
        #         if os.path.exists(video_path):
        #             converter.extract_frames_from_video(video_path, args.frame_root, video_name)
        #         else:
        #             print(f"   ⚠️  Skipping frame extraction: {video_path} does not exist")
        
        # Convert annotations
        print(f"\n📊 Converting annotations to COCO format...")
        converter.convert_batch(
            args.input,
            args.output,
            args.frame_step,
            args.width,
            args.height,
            generate_csv=args.csv,
            csv_output_path=args.csv_output,
            frame_root=args.frame_root,
            video_root=args.video_root
        )
        
    else:


        print(f"\n📊 Converting annotation to COCO format...")
        coco_data, sorted_frames = converter.convert_annotation(
            args.input,
            Path(args.input).stem,
            args.output,
            args.frame_step,
            args.width,
            args.height,
            args.frame_root
        )

        # Generate CSV if requested
        if args.csv:
            video_name = Path(args.input).stem
            converter.generate_csv_from_coco(coco_data, video_name, args.csv_output or f"{args.output}.csv")
        
        # Extract frames if requested
        if args.extract_frames:
            video_name = Path(args.input).stem
            video_path = os.path.join(args.video_root, f"{video_name}.mp4")
            print(f"🎬 Extracting frames...")
            if os.path.exists(video_path):
                converter.extract_frames_from_video(video_path, args.frame_root, video_name, sorted_frames=sorted_frames)
            else:
                print(f"⚠️  Skipping frame extraction: {video_path} does not exist")
        
    print("\n✅ Conversion complete!")
    import subprocess
    subprocess.run(["python", "merge_json.py", "--output-dir", args.output])  # Debugging line to show output directory contents
        