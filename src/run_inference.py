import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
import os
import sys

# Try to import ultralytics if available
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# Configuration
MODEL_PATH = r'E:\Suger_beets\runs\detect\train\weights\best.pt'
IMAGES_DIR = Path(r'E:\Suger_beets\data\test')
OUTPUT_DIR = Path(r'E:\Suger_beets\outputs\predictions')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_model(model_path):
    """Load YOLOv12 model from checkpoint"""
    print(f"Loading model from {model_path}...")
    try:
        # Add the required class to safe globals for PyTorch 2.6+
        import torch.serialization
        try:
            # Try to import the ultralytics module first
            import sys
            sys.path.append(".")  # Add current directory to path
            
            # Try different possible import paths for ultralytics
            try:
                from ultralytics.nn.tasks import DetectionModel
                torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
            except ImportError:
                pass
                
            # Load with weights_only=False for compatibility
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except Exception:
            # Fallback: try loading with weights_only=False only
            print("Attempting fallback loading method...")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Check if it's the full checkpoint or just the model
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model = checkpoint['model']
        else:
            model = checkpoint
            
        # Make sure it's in eval mode
        model.eval()
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTRY THIS ALTERNATIVE LOADING METHOD:")
        print("1. Install ultralytics package: pip install ultralytics")
        print("2. Use the following code instead:")
        print("   from ultralytics import YOLO")
        print("   model = YOLO('E:\\Suger_beets\\runs\\detect\\train\\weights\\best.pt')")
        print("   results = model('E:\\Suger_beets\\data\\test\\your_image.jpg')")
        return None

def preprocess_image(img_path):
    """Preprocess image for YOLOv12 inference"""
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to read image: {img_path}")
        return None, None
    
    # Get original dimensions for later
    orig_img = img.copy()
    
    # Resize and pad image to the required input size (assuming 640x640)
    input_size = 640
    h, w = img.shape[:2]
    scale = min(input_size / h, input_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize
    img = cv2.resize(img, (new_w, new_h))
    
    # Create padding
    pad_h, pad_w = input_size - new_h, input_size - new_w
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    
    # Apply padding
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize and convert to tensor
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    return img_tensor, orig_img

def postprocess_detections(predictions, img, orig_img, conf_threshold=0.25, iou_threshold=0.45):
    """Process model predictions and draw bounding boxes"""
    # Get height and width
    oh, ow = orig_img.shape[:2]
    
    # Convert predictions to numpy for easier handling
    if isinstance(predictions, (list, tuple)):
        # YOLO models typically return detection results as the first element
        predictions = predictions[0]
    
    # If predictions is a tensor, convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    
    # Extract boxes, confidence scores, and class ids
    # Adjust the slicing based on your model's output format
    # Typical format: [x1, y1, x2, y2, confidence, class_id]
    if len(predictions.shape) == 3:
        predictions = predictions[0]  # Get the first batch
    
    # Create a copy of the original image to draw on
    result_img = orig_img.copy()
    
    # Filter by confidence
    mask = predictions[:, 4] > conf_threshold
    filtered_predictions = predictions[mask]
    
    # Apply non-maximum suppression (implement if needed)
    # This is a simplified detection visualization
    boxes = []
    for det in filtered_predictions:
        x1, y1, x2, y2, conf, cls_id = det[:6]
        # Convert normalized coordinates to pixel coordinates
        x1, x2 = x1 * ow, x2 * ow
        y1, y2 = y1 * oh, y2 * oh
        
        # Add to the list of boxes
        boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_id)))
        
        # Draw bounding box
        cv2.rectangle(result_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Add label
        label = f"Class {int(cls_id)}: {conf:.2f}"
        cv2.putText(result_img, label, (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result_img, boxes

def run_inference():
    """Run inference on all images in the test directory"""
    # Load model
    model = load_model(MODEL_PATH)
    if model is None:
        return
    
    # Get list of image files
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    img_paths = [p for p in IMAGES_DIR.glob('*.*') if p.suffix.lower() in img_extensions]
    
    if not img_paths:
        print(f"No images found in {IMAGES_DIR}")
        return
    
    print(f"Found {len(img_paths)} images for inference")
    
    # Process each image
    for i, img_path in enumerate(img_paths):
        print(f"Processing image {i+1}/{len(img_paths)}: {img_path.name}")
        
        # Preprocess image
        img_tensor, orig_img = preprocess_image(img_path)
        if img_tensor is None:
            continue
        
        try:
            # Run inference
            start_time = time.time()
            with torch.no_grad():
                predictions = model(img_tensor)
            inference_time = time.time() - start_time
            print(f"Inference time: {inference_time:.3f} seconds")
            
            # Postprocess detections
            result_img, boxes = postprocess_detections(predictions, img_tensor[0].cpu().numpy(), orig_img)
            
            # Save result
            output_path = OUTPUT_DIR / f"{img_path.stem}_prediction{img_path.suffix}"
            cv2.imwrite(str(output_path), result_img)
            print(f"Saved results to {output_path}")
            
            # Print detection summary
            print(f"Found {len(boxes)} objects")
            
        except Exception as e:
            print(f"Error during inference on {img_path.name}: {e}")
    
    print("✅ Inference complete.")

def run_inference_with_ultralytics():
    """Run inference using ultralytics YOLO API (alternative method)"""
    if not ULTRALYTICS_AVAILABLE:
        print("Ultralytics package not found. Install with: pip install ultralytics")
        return
    
    print("Running inference using ultralytics YOLO API...")
    
    # Load the model
    try:
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model with ultralytics: {e}")
        return
    
    # Get list of image files
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    img_paths = [p for p in IMAGES_DIR.glob('*.*') if p.suffix.lower() in img_extensions]
    
    if not img_paths:
        print(f"No images found in {IMAGES_DIR}")
        return
    
    print(f"Found {len(img_paths)} images for inference")
    
    # Process each image
    for i, img_path in enumerate(img_paths):
        print(f"Processing image {i+1}/{len(img_paths)}: {img_path.name}")
        
        try:
            # Run inference
            results = model(str(img_path))
            
            # Save results
            result_path = OUTPUT_DIR / f"{img_path.stem}_prediction{img_path.suffix}"
            
            # Extract result image with annotations and save it
            result_img = results[0].plot()
            cv2.imwrite(str(result_path), result_img)
            
            print(f"Saved results to {result_path}")
            print(f"Found {len(results[0].boxes)} objects")
            
        except Exception as e:
            print(f"Error during inference on {img_path.name}: {e}")
    
    print("✅ Inference complete with ultralytics.")

if __name__ == "__main__":
    # Try the direct loading method first
    try:
        run_inference()
    except Exception as e:
        print(f"Direct inference failed: {e}")
        print("\nTrying alternative method with ultralytics...")
        
        # If direct loading fails, try using ultralytics
        if ULTRALYTICS_AVAILABLE:
            run_inference_with_ultralytics()
        else:
            print("\nPlease install ultralytics to use the alternative method:")
            print("pip install ultralytics")