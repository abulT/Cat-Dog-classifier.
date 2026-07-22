import torch
from torchvision import transforms
from PIL import Image
from model import CatDogCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CatDogCNN().to(device)
model.load_state_dict(torch.load("cat_dog_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

image_path = "test.jpg"   # Change this to your image name
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

classes = ["cats", "dogs"]

with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

print("Prediction:", classes[predicted.item()])