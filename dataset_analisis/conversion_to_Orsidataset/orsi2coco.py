import json
import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

# FPS configuration - adjust based on your video fps
FPS = 25  # Change this to match your video frame rate
INSTANT_EVENT_DURATION = 3  # seconds to keep instant events active (default 3)

# Try to import PIL for reading image dimensions
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class ORSI2COCO:
    """
    Convert ORSI dataset format (RARP01.json) to COCO format compatible with GraSP

    Source format: RARP01.json with EVENTS (steps) and PHASES
    Target format: grasp_long-term_train.json (COCO format)

    Event types:
    - Start/End events: paired events that define a time range
    - Instant events: single point in time events active for X seconds
    """

    def __init__(self, fps: int = FPS, instant_event_duration: float = INSTANT_EVENT_DURATION):
        self.fps = fps
        self.instant_event_duration = instant_event_duration
        self.image_id_counter = 1
        self.annotation_id_counter = 1
        self.phase_to_id = {}
        self.step_to_id = {}
        self.annotated_frames = set()  # Track frames with annotations
        self.dimensions_cache = {}  # Cache for image dimensions

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
        Identify which events are instant (point-in-time) vs range events

        Events are considered instant if:
        - They have few timestamps (typically 1)
        - They are not clearly part of a start/end pair
        """
        instant_events = set()

        for category_name, events in events_data.items():
            for event_name, timestamps in events.items():
                # Check if this event has a corresponding start/end pair
                has_end_event = False

                # Look for complementary end events
                for other_event_name in events.keys():
                    if event_name != other_event_name:
                        # Check if one is start and other is end
                        if "start" in event_name.lower() or "begin" in event_name.lower():
                            # This is a start event, check if there's an end
                            end_name = other_event_name.lower()
                            if "end" in end_name or (
                                event_name.lower().replace("start", "end") == end_name
                                or event_name.lower().replace("begin", "end") == end_name
                            ):
                                has_end_event = True
                                break

                # If no clear start/end pattern, it's an instant event
                if not has_end_event:
                    instant_events.add(event_name)

        return instant_events

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

    def get_frame_step(self, frame_num: int, events_data: Dict, instant_events: Set[str]) -> int:
        """
        Determine which step/event a frame belongs to based on timing

        For instant events: the event is active for instant_event_duration seconds
        For non-instant events: use the most recent event before this frame
        """
        frame_time = frame_num / self.fps

        # First, check for instant events that are active at this time
        for category_name in events_data.keys():
            for event_name, timestamps in events_data[category_name].items():
                if event_name in instant_events:
                    # Check if any instant event is active at this frame
                    for timestamp in timestamps:
                        event_start = timestamp
                        event_end = timestamp + self.instant_event_duration
                        if event_start <= frame_time <= event_end:
                            return self.step_to_id.get(event_name, 0)

        # If no instant event, find the most recent non-instant event
        closest_event = None
        closest_time = -float('inf')

        for category_name in events_data.keys():
            for event_name, timestamps in events_data[category_name].items():
                if event_name not in instant_events:
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
        instant_events = self.identify_instant_events(events_data)

        # Build COCO structure
        coco_output = {
            "info": {
                "description": "ORSI Dataset",
                "url": "https://example.com",
                "version": "1",
                "year": "2024",
                "contributor": "ORSI"
            },
            "phases_categories": self.build_phase_categories(phases_data),
            "steps_categories": self.build_step_categories(events_data),
            "images": [],
            "annotations": []
        }

        # Determine total duration
        max_time = 0
        for category in events_data.values():
            for timestamps in category.values():
                if timestamps:
                    max_time = max(max_time, max(timestamps))

        for category in phases_data.values():
            end_times = category.get("END", [])
            if end_times:
                max_time = max(max_time, max(end_times))

        total_frames = self.seconds_to_frame(max_time)

        # Collect all frames that have annotations (either from events or phases)
        frames_with_annotations = set()

        # Add frames from instant events
        for category in events_data.values():
            for event_name, timestamps in category.items():
                if event_name in instant_events:
                    for timestamp in timestamps:
                        # Add frames during the instant event duration
                        start_frame = self.seconds_to_frame(timestamp)
                        end_frame = self.seconds_to_frame(timestamp + self.instant_event_duration)
                        for f in range(start_frame, end_frame + 1, frame_step):
                            frames_with_annotations.add(f)

        # Add frames from non-instant events
        for category in events_data.values():
            for event_name, timestamps in category.items():
                if event_name not in instant_events:
                    for timestamp in timestamps:
                        # Add frames during the instant event duration
                        start_frame = self.seconds_to_frame(timestamp)
                        end_frame = self.seconds_to_frame(timestamp + self.instant_event_duration)
                        for f in range(start_frame, end_frame + 1, frame_step):
                            frames_with_annotations.add(f)

        # Add frames from phases
        for phase_name, phase_timing in phases_data.items():
            start_times = phase_timing.get("START", [])
            end_times = phase_timing.get("END", [])
            if start_times and end_times:
                start_frame = self.seconds_to_frame(start_times[0])
                end_frame = self.seconds_to_frame(end_times[0])
                frames_with_annotations.add(start_frame)
                frames_with_annotations.add(end_frame)

        # Generate frames: combine regular sampling with annotation frames
        all_frames = set()

        # Add regularly sampled frames
        for frame_num in range(0, total_frames, frame_step):
            all_frames.add(frame_num)

        # Add all frames with annotations
        all_frames.update(frames_with_annotations)

        # Sort frames
        sorted_frames = sorted(list(all_frames))

        # Create images and annotations
        for frame_num in sorted_frames:
            if frame_num < 0 or frame_num > total_frames:
                continue

            image_entry = self.create_image_entry(video_name, frame_num, width, height, frame_root)
            coco_output["images"].append(image_entry)

            # Get phase and step for this frame
            phase_id = self.get_frame_phase(frame_num, phases_data)
            step_id = self.get_frame_step(frame_num, events_data, instant_events)

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
        print(f"  Total frames: {total_frames}")
        print(f"  Extracted frames: {len(coco_output['images'])}")
        print(f"  Frames with annotations: {len(frames_with_annotations)}")
        print(f"  Instant events: {len(instant_events)}")
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
                     frame_root: Optional[str] = None):
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
    import argparse

    parser = argparse.ArgumentParser(description="Convert ORSI dataset to COCO format")
    parser.add_argument("--input", type=str, help="Input JSON file or directory with JSON files")
    parser.add_argument("--output", type=str, help="Output JSON file or directory")
    parser.add_argument("--fps", type=int, default=25, help="Video FPS (default: 25)")
    parser.add_argument("--frame-step", type=int, default=46, help="Extract every N frames (default: 46)")
    parser.add_argument("--width", type=int, default=1280, help="Video width (default: 1280)")
    parser.add_argument("--height", type=int, default=800, help="Video height (default: 800)")
    parser.add_argument("--instant-duration", type=float, default=3.0, help="Duration for instant events in seconds (default: 3.0)")
    parser.add_argument("--csv", action="store_true", help="Generate CSV files")
    parser.add_argument("--csv-output", type=str, help="Path for combined CSV file (default: output_dir/train.csv)")
    parser.add_argument("--frame-root", type=str, help="Root directory for frames to read actual image dimensions")
    parser.add_argument("--batch", action="store_true", help="Process all RARP*.json files in input directory")

    args = parser.parse_args()

    converter = ORSI2COCO(fps=args.fps, instant_event_duration=args.instant_duration)

    if args.batch:
        if not args.input or not args.output:
            print("--batch mode requires both --input and --output directories")
            exit(1)
        converter.convert_batch(
            args.input,
            args.output,
            args.frame_step,
            args.width,
            args.height,
            generate_csv=args.csv,
            csv_output_path=args.csv_output,
            frame_root=args.frame_root
        )
    else:
        if not args.input or not args.output:
            print("--input and --output are required")
            exit(1)
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
