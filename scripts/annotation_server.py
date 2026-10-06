from flask import Flask, request, jsonify, send_from_directory, render_template_string
import os
import json
import pandas as pd

app = Flask(__name__)

DATA_DIR = r"c:\projects\spine\data\vindr_vertebral_annotations"
IMG_DIR = os.path.join(DATA_DIR, "images")
ANN_FILE = os.path.join(DATA_DIR, "raw_annotations.json")

# Ensure dirs exist
os.makedirs(IMG_DIR, exist_ok=True)
if not os.path.exists(ANN_FILE):
    with open(ANN_FILE, "w") as f:
        json.dump([], f)

@app.route('/')
def index():
    # Read the images
    df = pd.read_csv(os.path.join(DATA_DIR, "selected_images.csv"))
    images = df['image_id'].tolist()
    
    with open(r"c:\projects\spine\scripts\annotation_tool.html", "r") as f:
        html = f.read()
    return render_template_string(html, images=images)

@app.route('/image/<img_id>')
def serve_image(img_id):
    return send_from_directory(IMG_DIR, f"{img_id}.png")

@app.route('/api/annotations', methods=['GET'])
def get_annotations():
    with open(ANN_FILE, "r") as f:
        data = json.load(f)
    return jsonify(data)

@app.route('/api/annotations', methods=['POST'])
def save_annotations():
    new_data = request.json
    
    # Read existing
    with open(ANN_FILE, "r") as f:
        data = json.load(f)
        
    # Overwrite if exists
    filtered = [d for d in data if d['image_id'] != new_data['image_id']]
    filtered.append(new_data)
    
    with open(ANN_FILE, "w") as f:
        json.dump(filtered, f, indent=4)
        
    return jsonify({"status": "success"})

if __name__ == '__main__':
    print("Starting Annotation Server on http://localhost:5000")
    print("Please open this link in your browser to annotate the images.")
    app.run(port=5000)
