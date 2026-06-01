# Xhesika Gjyla, Student ID: K12539281

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from shiny import reactive
from shiny.express import input, render, ui


class SatelliteCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


CLASS_NAMES = {
    0: "HerbaceousVegetation",
    1: "AnnualCrop",
    2: "Residential",
    3: "Pasture",
    4: "Industrial",
    5: "River",
    6: "Highway",
    7: "Forest",
    8: "PermanentCrop",
    9: "SeaLake"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(__file__).parent / "assets" / "weights" / "best_model.pth"

model = SatelliteCNN(num_classes=10).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
_ = model.load_state_dict(state_dict)
_ = model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@reactive.calc
def prediction_result():
    uploaded = input.image_file()

    if uploaded is None:
        return None

    image_path = uploaded[0]["datapath"]
    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    predicted_idx = torch.argmax(probabilities).item()
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = probabilities[predicted_idx].item()

    probability_df = pd.DataFrame({
        "Class": [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))],
        "Probability": [round(probabilities[i].item(), 4) for i in range(len(CLASS_NAMES))]
    })

    probability_df = probability_df.sort_values(
        by="Probability",
        ascending=False
    ).reset_index(drop=True)

    return {
        "image_path": image_path,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probability_df": probability_df
    }

ui.page_opts(title="Satellite Image Classifier", fillable=True)

with ui.sidebar():
    ui.h3("Upload image")

    ui.input_file(
        "image_file",
        "Select satellite image",
        accept=[".jpg", ".jpeg", ".png"],
        multiple=False
    )

    ui.input_action_button("predict_btn", "Predict class", class_="btn-outline-dark w-100")


with ui.layout_columns(col_widths=[6, 6]):

    with ui.card():
        ui.card_header("Image")

        @render.image
        def uploaded_image():
            uploaded = input.image_file()

            if uploaded is None:
                return None

            return {
                "src": uploaded[0]["datapath"],
                "width": "300px",
                "alt": "Uploaded image"
            }

    with ui.card():
        ui.card_header("Class Probabilities")

        @render.data_frame
        def probability_table():
            result = prediction_result()

            if result is None:
                return pd.DataFrame(columns=["Class", "Probability"])

            return render.DataTable(result["probability_df"])

with ui.card(style="height: 150px;"):
    ui.card_header("Prediction")

    @render.text
    @reactive.event(input.predict_btn)
    def prediction_text():
        result = prediction_result()

        if result is None:
            return "Please upload an image first."

        return (
            f"Predicted Class: {result['predicted_class']} "
            f"(Confidence: {result['confidence']:.2%})"
        )