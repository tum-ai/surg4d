#!/usr/bin/env python3
"""
Generate review images with spatial labels marked and annotated.

This script creates annotated images for surgeon review by:
1. Loading frames that have spatial labels
2. Drawing markers at the labeled pixel positions
3. Adding text annotations with the query/prompt
4. Saving to a flat folder structure for easy review

Usage:
    pixi run python generate_review_images.py
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse


def draw_marker(draw, x, y, color, letter, size=15):
    """Draw a crosshair marker with a letter at the specified position."""
    # Draw crosshair
    line_length = size
    draw.line([(x - line_length, y), (x + line_length, y)], fill=color, width=3)
    draw.line([(x, y - line_length), (x, y + line_length)], fill=color, width=3)
    
    # Draw circle
    draw.ellipse(
        [(x - size, y - size), (x + size, y + size)],
        outline=color,
        width=3
    )
    
    # Draw letter offset from marker (top-right)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Position text offset from the marker (above and to the right)
    text_offset_x = size + 8
    text_offset_y = -size - 8
    
    # Draw text with white outline for visibility
    text_x = x + text_offset_x
    text_y = y + text_offset_y
    
    # Draw outline/shadow for better visibility
    for offset in [(-1, -1), (-1, 1), (1, -1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]:
        draw.text(
            (text_x + offset[0], text_y + offset[1]),
            letter,
            fill='white',
            font=font
        )
    
    # Draw main text
    draw.text(
        (text_x, text_y),
        letter,
        fill=color,
        font=font
    )


def add_label_legend(draw, objects, actions, legend_width, img_height):
    """Add a legend box with all labels listed in the left padding area."""
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
        font_bold = ImageFont.load_default()
    
    # Prepare legend text
    legend_lines = []
    
    if objects:
        legend_lines.append(("OBJECTS:", "#2563eb", font_bold))
        for i, obj in enumerate(objects):
            # Adjust pixel coordinates to account for left padding
            original_x = obj['pixel_x']
            original_y = obj['pixel_y']
            text = f"  O{i+1}: {obj['query']}"
            legend_lines.append((text, "#2563eb", font))
            coord_text = f"      @ ({original_x}, {original_y})"
            legend_lines.append((coord_text, "#666666", font))
    
    if actions:
        if legend_lines:
            legend_lines.append(("", "#000000", font))  # Empty line
        legend_lines.append(("ACTIONS:", "#dc2626", font_bold))
        for i, action in enumerate(actions):
            # Adjust pixel coordinates to account for left padding
            original_x = action['pixel_x']
            original_y = action['pixel_y']
            text = f"  A{i+1}: {action['query']}"
            legend_lines.append((text, "#dc2626", font))
            coord_text = f"      @ ({original_x}, {original_y})"
            legend_lines.append((coord_text, "#666666", font))
    
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
            draw.text(
                (box_x + padding, y_offset),
                text,
                fill=color,
                font=text_font
            )
        y_offset += line_height


def process_scene(label_file, clips_dir, output_dir, category):
    """Process all labeled frames for a single scene."""
    scene_name = label_file.stem.replace('_spatial', '')
    
    # Load labels
    with open(label_file, 'r') as f:
        labels = json.load(f)
    
    # Find the clip directory
    clip_dir = clips_dir / category / scene_name
    if not clip_dir.exists():
        print(f"Warning: Clip directory not found: {clip_dir}")
        return 0
    
    images_created = 0
    
    # Left padding for legend
    legend_width = 500
    
    # Process each labeled frame
    for timestep, frame_data in labels.items():
        video_id = frame_data.get('video_id', scene_name)
        frame_number = frame_data.get('frame_number')
        objects = frame_data.get('objects', [])
        actions = frame_data.get('actions', [])
        
        # Skip frames with no labels
        if not objects and not actions:
            continue
        
        # Find the frame file
        frame_file = clip_dir / f"frame_{frame_number}_endo.png"
        if not frame_file.exists():
            print(f"Warning: Frame file not found: {frame_file}")
            continue
        
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
        
        # Draw object markers (blue) - adjust x coordinate by legend_width
        for i, obj in enumerate(objects):
            x, y = obj['pixel_x'] + legend_width, obj['pixel_y']
            draw_marker(draw, x, y, '#2563eb', f'O{i+1}')
        
        # Draw action markers (red) - adjust x coordinate by legend_width
        for i, action in enumerate(actions):
            x, y = action['pixel_x'] + legend_width, action['pixel_y']
            draw_marker(draw, x, y, '#dc2626', f'A{i+1}')
        
        # Add legend in the left padding area
        add_label_legend(draw, objects, actions, legend_width, new_height)
        
        # Composite overlay onto image
        padded_img = Image.alpha_composite(padded_img, overlay)
        
        # Convert back to RGB for saving
        padded_img = padded_img.convert('RGB')
        
        # Save with descriptive filename
        output_filename = f"{video_id}_frame{frame_number:05d}_review.jpg"
        output_path = output_dir / output_filename
        padded_img.save(output_path, 'JPEG', quality=95)
        
        images_created += 1
        print(f"  ✓ Created: {output_filename}")
    
    return images_created


def main():
    parser = argparse.ArgumentParser(
        description='Generate review images with spatial labels marked'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/review_images'),
        help='Output directory for review images (default: data/review_images)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    repo_root = Path(__file__).parent.parent
    labels_dir = repo_root / "data" / "labels"
    clips_dir = repo_root / "labeling" / "clips"
    output_dir = repo_root / args.output_dir
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GENERATING REVIEW IMAGES")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print()
    
    total_images = 0
    
    # Process both categories
    for category in ['seg80_t50_intersection', 'seg80_only']:
        category_label_dir = labels_dir / category
        
        if not category_label_dir.exists():
            continue
        
        print(f"\nProcessing category: {category}")
        print("-" * 80)
        
        # Process each scene
        for label_file in sorted(category_label_dir.glob('*_spatial.json')):
            scene_name = label_file.stem.replace('_spatial', '')
            print(f"\n{scene_name}:")
            
            images_created = process_scene(
                label_file, clips_dir, output_dir, category
            )
            total_images += images_created
    
    print()
    print("=" * 80)
    print(f"✓ COMPLETE: Generated {total_images} review images")
    print(f"✓ Location: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()


