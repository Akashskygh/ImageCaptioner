from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import matplotlib.pyplot as plt
import os

MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"

# Initializing the VisionEncoderDecoderModel, ViTImageProcessor, and AutoTokenizer
def initialize_model(model_name):
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model, ViTImageProcessor.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name)

# Loading images from given image paths
def load_images(image_paths):
    return [Image.open(image_path).convert(mode="RGB") for image_path in image_paths if os.path.isfile(image_path)]

# Generating captions using the provided model
def generate_captions(model, image_processor, tokenizer, device, images, batch_size=8, max_length=16, num_beams=4):
    preds = []
    for i in range(0, len(images), batch_size):
        pixel_values = image_processor(images=images[i:i + batch_size], return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            output_ids = model.generate(pixel_values, max_length=max_length, num_beams=num_beams)
        preds.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
    return preds

# Formatting caption to capitalize the first letter
def format_caption(caption, capitalize_first=True):
    return caption.capitalize() if capitalize_first else caption

# Predicting captions for images in a given list of image paths
def predict_captions(image_paths, model, image_processor, tokenizer, device, **kwargs):
    images = load_images(image_paths)
    captions = generate_captions(model, image_processor, tokenizer, device, images, **kwargs)
    return captions, images

# Displaying image with its caption
def display_image_with_caption(image_path, caption):
    plt.imshow(Image.open(image_path))
    plt.axis('off')
    plt.title(caption)
    plt.show()

# Getting image paths from a specified folder
def get_image_paths(folder_path):
    return [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Generating captions for all images in a folder
def generate_captions_for_folder(folder_path, model, image_processor, tokenizer, device, **kwargs):
    captions = {}
    for image_path in get_image_paths(folder_path):
        image_captions, _ = predict_captions([image_path], model, image_processor, tokenizer, device, **kwargs)
        captions[image_path] = image_captions[0] if image_captions else "No caption generated"
    return captions

# Initializing the model, image processor, and tokenizer
model, image_processor, tokenizer = initialize_model(MODEL_NAME)

# Specifying the folder containing images
imgs_folder = 'imgs'

# Generating captions for images in the specified folder
captions_for_imgs = generate_captions_for_folder(imgs_folder, model, image_processor, tokenizer, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Displaying formatted captions along with respective images
for image_path, caption in captions_for_imgs.items():
    formatted_caption = format_caption(caption, capitalize_first=True)
    print(f"Generated Caption for {image_path}:", formatted_caption)
    display_image_with_caption(image_path, formatted_caption)