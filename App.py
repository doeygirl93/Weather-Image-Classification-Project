# OMG IM ACTAULLY SO HAPPY TO BE WORKING IN PURE PYTHON AGAIN

import gradio as gr
import torch
import torch.nn as nn

from torchvision import transforms, datasets
from PIL import Image
from pathlib import Path

###from model_code import WeatherClassifierCNN, NUM_CLASSES, CLASS_NAMES

######################################


BASE_DIR = "Multi-class Weather Dataset"
CLOUDY_DIR = "Multi-class Weather Dataset/Cloudy"
RAIN_DIR = "Multi-class Weather Dataset/Rain"
SHINE_DIR = "Multi-class Weather Dataset/Shine"
SUNRISE_DIR = "Multi-class Weather Dataset/Sunrise"
FULL_dataset = datasets.ImageFolder(BASE_DIR, transform=None)



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size=3, pool=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernal_size, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool is not None:
            x = self.pool(x)
        return x
class WeatherClassifierCNN(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super(WeatherClassifierCNN, self).__init__()
        self.features = nn.Sequential(
            ConvBlock(input_channels, 64, pool=True),
            ConvBlock(64, 128, pool=True),
            ConvBlock(128, 256, pool=True),
            ConvBlock(256, 512, pool=True),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        #classfier part

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x







NUM_CLASSES = len(FULL_dataset.classes)
CLASS_NAMES = FULL_dataset.classes

#########################################


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

CLASS_NAMES = FULL_dataset.classes


model = WeatherClassifierCNN(num_classes=NUM_CLASSES).to(DEVICE)


SAVE_DIR = "checkpoints"
checkpoint =  torch.load(Path(SAVE_DIR) / 'best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
print(f'\n MODEL IS LOADED ---Epoch: {checkpoint['epoch']}, Accuracy : {checkpoint['accuracy']:.4}')
print(" Model is LOADED!!!!! Have fun with validation ]")

model.eval()


#PREDICTING FUN
def predict_weather(input_img):
    if input_img is None:
        return None

    ## Transformers

    #%%
    transformers = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Turn to tensor n shi
    img = transformers(input_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = nn.functional.softmax(outputs[0], dim=0)

    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

interface = gr.Interface(
    fn=predict_weather,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=len(CLASS_NAMES)),
    title = "AI for Weather Predition",
    description="Put in an image of the sky. Then the model will determine if the sky is Cloudy, Rainy, Sunrise, and Sunset."
)

interface.launch(share=True)