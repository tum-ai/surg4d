#!/usr/bin/env python3
"""
Spatial labeling tool for surgical scene frames.
Creates object and action labels with pixel coordinates.

Usage:
    pixi run python spatial_labeling_tool.py --clip video17_01563 --category seg80_only --port 5000

Then access via SSH port forwarding:
    ssh -L 5000:localhost:5000 user@remote
    Open browser to: http://localhost:5000
"""

import argparse
import json
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request, send_file
import base64

app = Flask(__name__)

# Global state
STATE = {
    'clip_name': None,
    'category': None,
    'clip_dir': None,
    'frames': [],
    'current_frame_idx': 0,
    'labels': {}
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Spatial Labeling Tool - {{ clip_name }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #333;
        }
        .title {
            font-size: 24px;
            font-weight: bold;
        }
        .progress {
            font-size: 18px;
            color: #666;
        }
        .main-content {
            display: flex;
            gap: 20px;
        }
        .image-section {
            flex: 2;
        }
        .canvas-container {
            position: relative;
            border: 2px solid #333;
            display: inline-block;
            cursor: crosshair;
        }
        canvas {
            display: block;
            max-width: 100%;
            height: auto;
        }
        .labels-section {
            flex: 1;
            min-width: 350px;
        }
        .label-type {
            margin-bottom: 20px;
            padding: 15px;
            background-color: #f8f8f8;
            border-radius: 5px;
        }
        .label-type h3 {
            margin-top: 0;
            color: #333;
        }
        .label-type.objects h3 {
            color: #2563eb;
        }
        .label-type.actions h3 {
            color: #dc2626;
        }
        .label-input {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }
        input[type="text"] {
            flex: 1;
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 14px;
        }
        button {
            padding: 8px 16px;
            background-color: #333;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }
        button:hover {
            background-color: #555;
        }
        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        .btn-primary {
            background-color: #2563eb;
        }
        .btn-primary:hover {
            background-color: #1d4ed8;
        }
        .btn-danger {
            background-color: #dc2626;
        }
        .btn-danger:hover {
            background-color: #b91c1c;
        }
        .btn-success {
            background-color: #16a34a;
        }
        .btn-success:hover {
            background-color: #15803d;
        }
        .label-list {
            margin-top: 15px;
        }
        .label-item {
            padding: 8px;
            margin-bottom: 8px;
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
        }
        .label-item.object {
            border-left: 4px solid #2563eb;
        }
        .label-item.action {
            border-left: 4px solid #dc2626;
        }
        .label-text {
            flex: 1;
            word-break: break-word;
        }
        .label-coords {
            color: #666;
            font-family: monospace;
            font-size: 12px;
            margin-right: 10px;
        }
        .delete-btn {
            padding: 4px 8px;
            font-size: 12px;
            background-color: #ef4444;
        }
        .delete-btn:hover {
            background-color: #dc2626;
        }
        .navigation {
            display: flex;
            gap: 10px;
            margin-top: 20px;
            justify-content: center;
        }
        .nav-btn {
            padding: 12px 24px;
            font-size: 16px;
        }
        .status {
            text-align: center;
            margin-top: 10px;
            padding: 10px;
            border-radius: 4px;
        }
        .status.success {
            background-color: #d1fae5;
            color: #065f46;
        }
        .status.error {
            background-color: #fee2e2;
            color: #991b1b;
        }
        .pending-coord {
            padding: 10px;
            background-color: #fef3c7;
            border: 1px solid #f59e0b;
            border-radius: 4px;
            margin-bottom: 10px;
            font-size: 14px;
        }
        .video-section {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f8f8;
            border-radius: 5px;
        }
        .video-section h3 {
            margin-top: 0;
        }
        video {
            width: 100%;
            max-width: 640px;
            border: 2px solid #333;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">Spatial Labeling Tool</div>
            <div class="progress">
                <strong>{{ clip_name }}</strong> - Frame <span id="frame-number">1</span> of <span id="total-frames">0</span>
            </div>
        </div>

        <div class="main-content">
            <div class="image-section">
                <div class="canvas-container">
                    <canvas id="canvas"></canvas>
                </div>
                
                <div class="navigation">
                    <button class="nav-btn" onclick="previousFrame()" id="prev-btn">← Previous</button>
                    <button class="nav-btn btn-success" onclick="saveAndNext()">Save & Next →</button>
                    <button class="nav-btn btn-primary" onclick="nextFrame()">Next →</button>
                </div>

                <div class="video-section">
                    <h3>Reference Video</h3>
                    <video id="reference-video" controls loop>
                        <source src="/video" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>

                <div id="status" class="status" style="display: none;"></div>
            </div>

            <div class="labels-section">
                <div class="label-type objects">
                    <h3>Object Labels</h3>
                    <div id="pending-object" class="pending-coord" style="display: none;">
                        Click on image to set coordinates
                    </div>
                    <div class="label-input">
                        <input type="text" id="object-query" placeholder="Object description (e.g., 'grasper tip')" />
                        <button class="btn-primary" onclick="startObjectLabel()">Add Object</button>
                    </div>
                    <div class="label-list" id="object-list"></div>
                </div>

                <div class="label-type actions">
                    <h3>Action Labels</h3>
                    <div id="pending-action" class="pending-coord" style="display: none;">
                        Click on image to set coordinates
                    </div>
                    <div class="label-input">
                        <input type="text" id="action-query" placeholder="Action description (e.g., 'cutting tissue')" />
                        <button class="btn-danger" onclick="startActionLabel()">Add Action</button>
                    </div>
                    <div class="label-list" id="action-list"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentFrame = 0;
        let frames = [];
        let labels = {};
        let pendingLabel = null;
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let currentImage = null;
        
        // Store clip name globally
        window.clipName = "{{ clip_name }}";

        // Initialize
        async function init() {
            await loadFrames();
            await loadLabels();
            loadFrame(0);
        }

        async function loadFrames() {
            const response = await fetch('/api/frames');
            frames = await response.json();
            document.getElementById('total-frames').textContent = frames.length;
        }

        async function loadLabels() {
            const response = await fetch('/api/labels');
            labels = await response.json();
        }

        async function loadFrame(index) {
            if (index < 0 || index >= frames.length) return;
            
            currentFrame = index;
            const frameName = frames[index];
            
            // Load image
            const response = await fetch(`/api/frame/${frameName}`);
            const blob = await response.blob();
            const img = new Image();
            img.onload = function() {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                currentImage = img;
                drawLabels();
            };
            img.src = URL.createObjectURL(blob);
            
            // Update UI
            document.getElementById('frame-number').textContent = index + 1;
            document.getElementById('prev-btn').disabled = (index === 0);
            
            // Update label lists
            updateLabelLists();
            clearPendingLabel();
        }

        function drawLabels() {
            if (!currentImage) return;
            
            // Redraw image
            ctx.drawImage(currentImage, 0, 0);
            
            // Draw existing labels using timestep index
            const timestep = currentFrame.toString();
            const frameLabels = labels[timestep] || { objects: [], actions: [] };
            
            // Draw object labels (blue)
            frameLabels.objects.forEach(label => {
                drawMarker(label.pixel_x, label.pixel_y, '#2563eb', 'O');
            });
            
            // Draw action labels (red)
            frameLabels.actions.forEach(label => {
                drawMarker(label.pixel_x, label.pixel_y, '#dc2626', 'A');
            });
        }

        function drawMarker(x, y, color, letter) {
            ctx.save();
            
            // Draw crosshair
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x - 10, y);
            ctx.lineTo(x + 10, y);
            ctx.moveTo(x, y - 10);
            ctx.lineTo(x, y + 10);
            ctx.stroke();
            
            // Draw circle
            ctx.beginPath();
            ctx.arc(x, y, 15, 0, 2 * Math.PI);
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Draw letter
            ctx.fillStyle = color;
            ctx.font = 'bold 14px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(letter, x, y);
            
            ctx.restore();
        }

        canvas.addEventListener('click', function(event) {
            if (!pendingLabel) return;
            
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const x = Math.round((event.clientX - rect.left) * scaleX);
            const y = Math.round((event.clientY - rect.top) * scaleY);
            
            completePendingLabel(x, y);
        });

        function startObjectLabel() {
            const query = document.getElementById('object-query').value.trim();
            if (!query) {
                alert('Please enter an object description');
                return;
            }
            
            pendingLabel = { type: 'object', query: query };
            document.getElementById('pending-object').style.display = 'block';
            document.getElementById('pending-object').textContent = `Click on image to mark: "${query}"`;
        }

        function startActionLabel() {
            const query = document.getElementById('action-query').value.trim();
            if (!query) {
                alert('Please enter an action description');
                return;
            }
            
            pendingLabel = { type: 'action', query: query };
            document.getElementById('pending-action').style.display = 'block';
            document.getElementById('pending-action').textContent = `Click on image to mark: "${query}"`;
        }

        function completePendingLabel(x, y) {
            const timestep = currentFrame.toString();
            const frameName = frames[currentFrame];
            
            // Extract frame number from filename (e.g., "frame_1563_endo.png" -> 1563)
            const frameNum = parseInt(frameName.match(/frame_(\d+)_endo\.png/)[1]);
            
            if (!labels[timestep]) {
                labels[timestep] = {
                    video_id: window.clipName,
                    frame_number: frameNum,
                    objects: [],
                    actions: []
                };
            }
            
            const label = {
                query: pendingLabel.query,
                pixel_x: x,
                pixel_y: y,
                // Also store as [y, x] for numpy compatibility
                pixel_coords_numpy: [y, x]
            };
            
            if (pendingLabel.type === 'object') {
                labels[timestep].objects.push(label);
                document.getElementById('object-query').value = '';
            } else {
                labels[timestep].actions.push(label);
                document.getElementById('action-query').value = '';
            }
            
            clearPendingLabel();
            drawLabels();
            updateLabelLists();
            showStatus('Label added successfully!', 'success');
        }

        function clearPendingLabel() {
            pendingLabel = null;
            document.getElementById('pending-object').style.display = 'none';
            document.getElementById('pending-action').style.display = 'none';
        }

        function updateLabelLists() {
            const timestep = currentFrame.toString();
            const frameLabels = labels[timestep] || { objects: [], actions: [] };
            
            // Update object list
            const objectList = document.getElementById('object-list');
            objectList.innerHTML = '';
            frameLabels.objects.forEach((label, index) => {
                const div = document.createElement('div');
                div.className = 'label-item object';
                div.innerHTML = `
                    <div class="label-text">${label.query}</div>
                    <div class="label-coords">(${label.pixel_x}, ${label.pixel_y})</div>
                    <button class="delete-btn" onclick="deleteLabel('object', ${index})">Delete</button>
                `;
                objectList.appendChild(div);
            });
            
            // Update action list
            const actionList = document.getElementById('action-list');
            actionList.innerHTML = '';
            frameLabels.actions.forEach((label, index) => {
                const div = document.createElement('div');
                div.className = 'label-item action';
                div.innerHTML = `
                    <div class="label-text">${label.query}</div>
                    <div class="label-coords">(${label.pixel_x}, ${label.pixel_y})</div>
                    <button class="delete-btn" onclick="deleteLabel('action', ${index})">Delete</button>
                `;
                actionList.appendChild(div);
            });
        }

        function deleteLabel(type, index) {
            const timestep = currentFrame.toString();
            if (!labels[timestep]) return;
            
            if (type === 'object') {
                labels[timestep].objects.splice(index, 1);
            } else {
                labels[timestep].actions.splice(index, 1);
            }
            
            drawLabels();
            updateLabelLists();
        }

        async function saveLabels() {
            const response = await fetch('/api/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(labels)
            });
            
            if (response.ok) {
                showStatus('Labels saved successfully!', 'success');
                return true;
            } else {
                showStatus('Error saving labels', 'error');
                return false;
            }
        }

        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + type;
            status.style.display = 'block';
            setTimeout(() => {
                status.style.display = 'none';
            }, 3000);
        }

        function previousFrame() {
            if (currentFrame > 0) {
                loadFrame(currentFrame - 1);
            }
        }

        function nextFrame() {
            if (currentFrame < frames.length - 1) {
                loadFrame(currentFrame + 1);
            }
        }

        async function saveAndNext() {
            const saved = await saveLabels();
            if (saved) {
                nextFrame();
            }
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', function(event) {
            if (event.key === 'ArrowLeft') {
                previousFrame();
            } else if (event.key === 'ArrowRight') {
                nextFrame();
            } else if (event.key === 's' && event.ctrlKey) {
                event.preventDefault();
                saveLabels();
            }
        });

        // Auto-save on unload
        window.addEventListener('beforeunload', function() {
            navigator.sendBeacon('/api/save', JSON.stringify(labels));
        });

        init();
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, clip_name=STATE['clip_name'])


@app.route('/api/frames')
def get_frames():
    """Get list of frame filenames"""
    return jsonify(STATE['frames'])


@app.route('/api/frame/<frame_name>')
def get_frame(frame_name):
    """Serve a specific frame image"""
    frame_path = STATE['clip_dir'] / frame_name
    if frame_path.exists():
        return send_file(frame_path, mimetype='image/png')
    return "Frame not found", 404


@app.route('/video')
def get_video():
    """Serve the reference video"""
    video_path = STATE['clip_dir'] / f"{STATE['clip_name']}.mp4"
    if video_path.exists():
        return send_file(video_path, mimetype='video/mp4')
    return "Video not found", 404


@app.route('/api/labels')
def get_labels():
    """Get all labels"""
    return jsonify(STATE['labels'])


@app.route('/api/save', methods=['POST'])
def save_labels():
    """Save labels to JSON file"""
    try:
        labels = request.json
        STATE['labels'] = labels
        
        # Save to labels directory (separate from clips)
        label_file = STATE['label_file']
        label_file.parent.mkdir(parents=True, exist_ok=True)
        with open(label_file, 'w') as f:
            json.dump(labels, f, indent=2)
        
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def load_existing_labels(label_file):
    """Load existing labels if they exist"""
    if label_file.exists():
        with open(label_file, 'r') as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description='Spatial labeling tool')
    parser.add_argument('--clip', required=True, help='Clip name (e.g., video17_01563)')
    parser.add_argument('--category', required=True, help='Category (seg80_only or seg80_t50_intersection)')
    parser.add_argument('--port', type=int, default=5000, help='Port to run server on')
    
    args = parser.parse_args()
    
    # Setup paths
    labeling_root = Path(__file__).parent
    repo_root = labeling_root.parent
    clip_dir = labeling_root / "clips" / args.category / args.clip
    label_file = repo_root / "data" / "labels" / args.category / f"{args.clip}_spatial.json"
    
    if not clip_dir.exists():
        print(f"Error: Clip directory not found: {clip_dir}")
        return
    
    # Get frame files (sorted by frame number)
    frames = sorted([f.name for f in clip_dir.glob("frame_*_endo.png")])
    
    if not frames:
        print(f"Error: No frames found in {clip_dir}")
        return
    
    # Initialize state
    STATE['clip_name'] = args.clip
    STATE['category'] = args.category
    STATE['clip_dir'] = clip_dir
    STATE['label_file'] = label_file
    STATE['frames'] = frames
    STATE['labels'] = load_existing_labels(label_file)
    
    print(f"\n{'='*60}")
    print(f"Spatial Labeling Tool")
    print(f"{'='*60}")
    print(f"Clip: {args.clip}")
    print(f"Category: {args.category}")
    print(f"Frames: {len(frames)}")
    print(f"Port: {args.port}")
    print(f"\nTo access the tool:")
    print(f"1. Set up SSH port forwarding:")
    print(f"   ssh -L {args.port}:localhost:{args.port} user@remote")
    print(f"2. Open in your browser:")
    print(f"   http://localhost:{args.port}")
    print(f"\nKeyboard shortcuts:")
    print(f"  - Left/Right arrows: Navigate frames")
    print(f"  - Ctrl+S: Save labels")
    print(f"\n{'='*60}\n")
    
    app.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == '__main__':
    main()

