#!/usr/bin/env python3
"""
Generate review images with temporal labels marked and annotated.

This script creates annotated images for surgeon review by:
1. Loading all frames (0-19) for each clip with temporal labels
2. Drawing annotations showing which temporal queries are active at each frame
3. Adding text annotations with the query/prompt in a side panel
4. Saving frames as 0.png, 1.png, ... 19.png in a directory per clip

Usage:
    pixi run python generate_review_images_temporal.py
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse


def is_frame_in_range(frame: int, ranges: list) -> bool:
    """Check if a frame is within any of the given ranges."""
    for range_pair in ranges:
        if len(range_pair) == 2:
            start, end = range_pair
            if start <= frame <= end:
                return True
    return False


def get_active_labels_for_frame(annotations: list, frame: int) -> list:
    """Get all labels that are active at the given frame."""
    active = []
    
    for ann in annotations:
        query_type = ann.get('query_type')
        ground_truth = ann.get('ground_truth', {})
        
        is_active = False
        
        if query_type == 'action_onset':
            # Check if this frame matches the onset
            onset_frame = ground_truth.get('frame')
            if onset_frame == frame:
                is_active = True
        
        elif query_type == 'action_duration':
            # Check if this frame is in any of the ranges
            ranges = ground_truth.get('ranges', [])
            if ranges and is_frame_in_range(frame, ranges):
                is_active = True
        
        if is_active:
            active.append(ann)
    
    return active


def add_label_legend(draw, active_labels, legend_width, img_height):
    """Add a legend box with all active labels listed in the left padding area."""
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
        font_bold = ImageFont.load_default()
    
    # Prepare legend text
    legend_lines = []
    
    if active_labels:
        legend_lines.append(("ACTIVE QUERIES:", "#dc2626", font_bold))
        for i, label in enumerate(active_labels):
            query_id = label.get('query_id', f'q{i+1}')
            query_type = label.get('query_type', 'unknown')
            question = label.get('question', '')
            
            # Extract query number from query_id (e.g., "video55_00588_q3" -> "q3")
            query_num = query_id.split('_q')[-1] if '_q' in query_id else str(i+1)
            
            type_label = "ONSET" if query_type == "action_onset" else "DURATION"
            text = f"  Q{query_num} [{type_label}]: {question}"
            legend_lines.append((text, "#dc2626", font))
    else:
        legend_lines.append(("No active queries", "#666666", font))
    
    if not legend_lines:
        return
    
    # Calculate legend box size
    padding = 15
    line_height = 24
    
    # Position legend in the left padding area
    margin = 15
    box_x = margin
    box_y = margin
    box_width = legend_width - 2 * margin
    
    # Calculate required height
    box_height = len(legend_lines) * line_height + 2 * padding
    
    # Ensure box doesn't exceed image height
    if box_y + box_height > img_height - margin:
        box_height = img_height - box_y - margin
    
    # Draw white background
    draw.rectangle(
        [(box_x, box_y), (box_x + box_width, box_y + box_height)],
        fill=(255, 255, 255, 255),
        outline="#333333",
        width=3
    )
    
    # Draw legend text
    y_offset = box_y + padding
    for text, color, text_font in legend_lines:
        if text:
            # Handle text wrapping if needed
            max_width = box_width - 2 * padding
            words = text.split(' ')
            lines = []
            current_line = []
            current_width = 0
            
            for word in words:
                # Approximate text width (rough estimate)
                word_width = len(word) * 9  # Rough estimate
                if current_width + word_width > max_width and current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_width = word_width
                else:
                    current_line.append(word)
                    current_width += word_width + 9  # Add space width
            
            if current_line:
                lines.append(' '.join(current_line))
            
            for line in lines:
                draw.text(
                    (box_x + padding, y_offset),
                    line,
                    fill=color,
                    font=text_font
                )
                y_offset += line_height
        else:
            y_offset += line_height


def process_clip(label_file, clips_dir, output_dir):
    """Process all frames for a single clip."""
    clip_name = label_file.stem.replace('_temporal', '')
    
    # Load labels
    with open(label_file, 'r') as f:
        labels = json.load(f)
    
    annotations = labels.get('annotations', [])
    num_frames = labels.get('clip_info', {}).get('num_frames', 20)
    
    # Find the clip directory
    clip_dir = clips_dir / clip_name
    if not clip_dir.exists():
        print(f"Warning: Clip directory not found: {clip_dir}")
        return 0
    
    # Create output directory for this clip
    clip_output_dir = output_dir / clip_name
    clip_output_dir.mkdir(parents=True, exist_ok=True)
    
    images_created = 0
    
    # Left padding for legend
    legend_width = 500
    
    # Process each frame (0 to num_frames-1)
    for frame_idx in range(num_frames):
        # Find the frame file (format: frame_XXX_endo_YY.png where YY is zero-padded frame_idx)
        frame_pattern = f"*_endo_{frame_idx:02d}.png"
        frame_files = list(clip_dir.glob(frame_pattern))
        
        if not frame_files:
            print(f"Warning: Frame file not found for frame {frame_idx} in {clip_dir}")
            continue
        
        frame_file = frame_files[0]  # Take the first match
        
        # Load original image
        original_img = Image.open(frame_file).convert('RGB')
        
        # Create new image with left padding
        new_width = original_img.width + legend_width
        new_height = original_img.height
        padded_img = Image.new('RGB', (new_width, new_height), color=(240, 240, 240))
        
        # Paste original image to the right of padding
        padded_img.paste(original_img, (legend_width, 0))
        
        # Convert to RGBA for drawing
        padded_img = padded_img.convert('RGBA')
        overlay = Image.new('RGBA', padded_img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Get active labels for this frame
        active_labels = get_active_labels_for_frame(annotations, frame_idx)
        
        # Add legend in the left padding area
        add_label_legend(draw, active_labels, legend_width, new_height)
        
        # Add frame number indicator
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Draw frame number in bottom-left of legend area
        frame_text = f"Frame {frame_idx}"
        draw.text(
            (legend_width // 2, new_height - 40),
            frame_text,
            fill='#000000',
            font=font,
            anchor='mm'
        )
        
        # Composite overlay onto image
        padded_img = Image.alpha_composite(padded_img, overlay)
        
        # Convert back to RGB for saving
        padded_img = padded_img.convert('RGB')
        
        # Save with simple frame number filename (2-digit padded)
        output_filename = f"{frame_idx:02d}.png"
        output_path = clip_output_dir / output_filename
        padded_img.save(output_path, 'PNG')
        
        images_created += 1
    
    return images_created


def main():
    parser = argparse.ArgumentParser(
        description='Generate review images with temporal labels marked'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/review_images_temporal'),
        help='Output directory for review images (default: data/review_images_temporal)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    repo_root = Path(__file__).parent.parent
    labels_dir = repo_root / "labeling" / "temporal_labels"
    clips_dir = repo_root / "labeling" / "clips"
    output_dir = repo_root / args.output_dir
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GENERATING TEMPORAL REVIEW IMAGES")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print()
    
    total_images = 0
    
    # Process each temporal label file
    for label_file in sorted(labels_dir.glob('*_temporal.json')):
        clip_name = label_file.stem.replace('_temporal', '')
        print(f"\n{clip_name}:")
        
        images_created = process_clip(
            label_file, clips_dir, output_dir
        )
        total_images += images_created
        print(f"  ✓ Created {images_created} frames")
    
    print()
    print("=" * 80)
    print(f"✓ COMPLETE: Generated {total_images} review images")
    print(f"✓ Location: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
