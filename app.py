from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from utils import load_model, predict_with_cam

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['CAM_FOLDER'] = 'static/cams'

# Class names per model
class_names_dict = {
    "cotton": ['Aphids', 'Army Worm', 'Bacterial Blight', 'Cotton Boll Rot', 'Green Cotton Boll', 'Healthy', 'Powdery Mildew', 'Target Spot'],

    "Cottonn": ['Bacterial Blight', 'Curl Virus', 'Healthy Leaf', 'Herbicide Growth Damage', 'Leaf Hopper Jassids', 'Leaf Reddening', 'Leaf Variegation'],

    "fabric_percent": ['30%', '40%', '50%', '53%', '58%', '60%', '63%', '65%', '66%', '68%', '90%', '98%', '100% Cotton'],

    "fabric_stain": ['Blood Spot', 'Coffee Stain', 'Detergent Stain', 'Food Spot', 'Glue Spot', 'Ink Stain', 'Makeup Stain',
                     'Marker Spot', 'Oil Stain', 'Paint Spot', 'Rust Stain', 'Sweat Stain']

}

# Load models
models = {
    "cotton": load_model("model/CottonLeafNet_levit_model.pth", len(class_names_dict["cotton"])),
    "Cottonn": load_model("model/SAR-CLD-2024_levit_model.pth", len(class_names_dict["Cottonn"])),
    "fabric_percent": load_model("model/CottonFabricImageBD_levit_model.pth", len(class_names_dict["fabric_percent"])),
    "fabric_stain": load_model("model/FabricSpotDefect_levit_model.pth", len(class_names_dict["fabric_stain"])),
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/<model_name>", methods=["GET", "POST"])
def predict(model_name):
    if model_name not in models:
        return "Model not found", 404

    prediction = None
    probabilities = []
    uploaded_image = ""
    cam_url = ""

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            cam_path = os.path.join(app.config['CAM_FOLDER'], filename)
            file.save(image_path)

            prediction, probabilities = predict_with_cam(
                models[model_name], image_path, cam_path, class_names_dict[model_name]
            )

            uploaded_image = filename
            cam_url = f"/static/cams/{filename}"

    return render_template(
        f"{model_name}.html",
        prediction=prediction,
        probabilities=probabilities,
        uploaded_image=uploaded_image,
        cam_url=cam_url,
        combined=zip(class_names_dict[model_name], probabilities)
    )

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['CAM_FOLDER'], exist_ok=True)
    app.run(debug=True)
