import torch
from PIL import Image
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import numpy as np

# Load BLIP image captioning model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load YOLOv8 via ultralytics (pip install ultralytics)
from ultralytics import YOLO

yolo_model = YOLO("yolov8n.pt")  # Use the nano version for fast inference

def describe_image(image_path):
    """Generate a detailed description of the image."""
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    caption_ids = model.generate(**inputs)
    caption = processor.decode(caption_ids[0], skip_special_tokens=True)
    return caption

def detect_objects(image_path):
    """Detect objects in the image and return their classes and bounding boxes."""
    results = yolo_model(image_path)
    data = results[0]
    objects = []
    for box in data.boxes:
        cls_id = int(box.cls[0])
        label = data.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        objects.append({'label': label, 'bbox': (x1, y1, x2, y2)})
    return objects

def extract_text(image_path):
    """Extract text from the image using OCR."""
    image = Image.open(image_path).convert('RGB')
    text = pytesseract.image_to_string(image)
    return text

def backward_search(image_path, query):
    """Search for objects or text matching the query in the image."""
    # Search in objects
    objects = detect_objects(image_path)
    found_objects = [obj for obj in objects if query.lower() in obj['label'].lower()]

    # Search in OCR text
    text = extract_text(image_path)
    found_text = query.lower() in text.lower()

    results = {
        'objects': found_objects,
        'text_found': found_text,
        'raw_text': text if found_text else ""
    }
    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Agent for Image Description and Backward Search")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--query", help="Query for backward search (object or text to find)", default=None)
    args = parser.parse_args()

    print("Generating description...")
    description = describe_image(args.image_path)
    print(f"Description: {description}")

    if args.query:
        print(f"\nPerforming backward search for: {args.query}")
        results = backward_search(args.image_path, args.query)
        if results['objects']:
            print(f"Found objects: {results['objects']}")
        if results['text_found']:
            print(f"Found text: {args.query} in image text.")
            print(f"OCR Text:\n{results['raw_text']}")
        if not results['objects'] and not results['text_found']:
            print("Query not found in objects or text.")