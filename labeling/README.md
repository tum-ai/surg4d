# Spatial Labeling Tool

This directory contains tools for labeling surgical scene data.

## Directory Structure

```
labeling/
├── clips/                          # Frame data (images and videos)
│   ├── seg80_only/
│   │   ├── video17_01563/
│   │   │   ├── frame_1563_endo.png
│   │   │   ├── frame_1567_endo.png
│   │   │   ├── ...
│   │   │   └── video17_01563.mp4
│   │   └── ...
│   └── seg80_t50_intersection/
│       └── ...
├── spatial_labeling_tool.py        # Web-based labeling interface
├── prepare_labeling_frames.py     # Script to prepare frames from dataset
└── README.md

data/
└── labels/                         # Label files (JSON, in data/ for easy git commit)
    ├── seg80_only/
    │   ├── video17_01563_spatial.json
    │   └── ...
    └── seg80_t50_intersection/
        └── ...
```

## Quick Start

### 1. Start the labeling tool

```bash
pixi run python spatial_labeling_tool.py --clip video17_01563 --category seg80_only --port 5000
```

### 2. Set up SSH port forwarding

In a new terminal on your local machine:
```bash
ssh -L 5000:localhost:5000 user@remote-server
```

### 3. Open in browser

Navigate to: `http://localhost:5000`

## Usage

### Interface Overview

- **Left Panel**: Shows the current frame with clickable canvas
- **Right Panel**: Tools for adding object and action labels
- **Reference Video**: Full clip video for context

### Adding Labels

1. **Object Labels** (e.g., "grasper tip", "gallbladder"):
   - Enter description in the object text field
   - Click "Add Object" button
   - Click on the image where the object is located

2. **Action Labels** (e.g., "cutting tissue", "grasping"):
   - Enter description in the action text field
   - Click "Add Action" button
   - Click on the image where the action is occurring

### Navigation

- **Previous/Next buttons**: Navigate through frames
- **Save & Next**: Save current labels and move to next frame
- **Arrow keys**: ← Previous frame, → Next frame
- **Ctrl+S**: Save labels

### Label Format

Labels are saved as `{clip_name}_spatial.json` in the labels directory.

The file uses 0-indexed timesteps (0-19 for 20 labeled frames), with each timestep containing:
- `video_id`: The clip identifier
- `frame_number`: The actual frame number from the video
- `objects`: List of object labels
- `actions`: List of action labels

```json
{
  "0": {
    "video_id": "video17_01563",
    "frame_number": 1563,
    "objects": [
      {
        "query": "grasper tip",
        "pixel_x": 425,
        "pixel_y": 320,
        "pixel_coords_numpy": [320, 425]
      }
    ],
    "actions": [
      {
        "query": "cutting tissue",
        "pixel_x": 512,
        "pixel_y": 380,
        "pixel_coords_numpy": [380, 512]
      }
    ]
  },
  "1": {
    "video_id": "video17_01563",
    "frame_number": 1567,
    "objects": [],
    "actions": []
  }
}
```

### Coordinate Format

- `pixel_x`, `pixel_y`: Standard (x, y) coordinates for PIL/image processing
- `pixel_coords_numpy`: [row, col] or [y, x] format for numpy arrays

## Tips

- Use the reference video to understand the full context of each frame
- Labels are auto-saved when navigating to prevent data loss
- You can delete labels by clicking the "Delete" button next to each label
- The visual markers show: Blue "O" for objects, Red "A" for actions

## Version Control

The `labeling/clips/` directory is gitignored (contains large image and video files), while labels are stored in `data/labels/` at the repository root and should be committed to track labeling progress. This separation makes it easy to:
- Commit only label files without large binary data
- Share labels without sharing the full dataset
- Track labeling progress in version control
- Keep labels organized with other data artifacts in the `data/` directory

