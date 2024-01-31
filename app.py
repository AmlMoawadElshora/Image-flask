from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

app = Flask(__name__)

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()
    return model

model = load_model()

def make_prediction(img): 
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))
    prediction = prediction[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction):
    img_tensor = torch.tensor(img)
    img_with_bboxes = draw_bounding_boxes(img_tensor, boxes=prediction["boxes"], labels=prediction["labels"],
                                          colors=["red" if label == "person" else "green" for label in prediction["labels"]], width=2)
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)
    return img_with_bboxes_np

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            img = Image.open(file)
            prediction = make_prediction(img)
            img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2, 0, 1), prediction)

            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(img_with_bbox)
            plt.xticks([], [])
            plt.yticks([], [])
            ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

            save_path = os.path.join("static", "result.png")
            plt.savefig(save_path)
            plt.close()

            del prediction["boxes"]
            return render_template("result.html", prediction=prediction)

    return render_template("index.html")

@app.route('/predictApi', methods=["POST"])
def api():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Please try again. The image does not exist.'})

        image = request.files.get('file')
        predictions = make_api_prediction(image)
        predicted_label = predictions["labels"][0] if predictions["labels"] else "N/A"
        return jsonify({'prediction': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)})

def make_api_prediction(image):
    img = Image.open(image)
    prediction = make_prediction(img)
    return prediction

if __name__ == "__main__":
    app.run(debug=True)
