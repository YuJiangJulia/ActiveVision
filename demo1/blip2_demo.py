import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import requests
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)

image = Image.open(requests.get("https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg", stream=True).raw).convert("RGB")

prompt = "Find the object used for writing"

inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
out = model.generate(**inputs)
print("🧠 Prediction:", processor.decode(out[0], skip_special_tokens=True))
