import os
import cv2
import numpy as np
import tensorflow as tf
import glob
import time

# --- CONFIGURATION ---
MODEL_PATH = 'drone_detector.tflite'
LABEL_PATH = 'labelmap.txt'
MIN_CONFIDENCE = 0.5        # Threshold (0.5 = 50%)
IMAGE_INPUT_DIR = 'test_images'   # Default folder for Image Mode
IMAGE_OUTPUT_DIR = 'output_images' # Folder to save results
# ---------------------

def load_model_and_labels():
    # 1. Load Labels
    with open(LABEL_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # 2. Load TFLite Model
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    float_input = (input_details[0]['dtype'] == np.float32)

    return interpreter, labels, input_details, output_details, width, height, float_input

def detect_objects(image, interpreter, labels, input_details, output_details, w, h, is_float):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (w, h))
    input_data = np.expand_dims(image_resized, axis=0)

    if is_float:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get Results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[3]['index'])[0]
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    imH, imW, _ = image.shape

    # Draw Boxes
    for i in range(len(scores)):
        if scores[i] > MIN_CONFIDENCE:
            # Convert box coords
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            # Draw Box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            # Draw Label
            class_idx = int(classes[i])
            label_name = labels[class_idx] if class_idx < len(labels) else f"Class {class_idx}"
            label_text = f'{label_name}: {int(scores[i]*100)}%'

            labelSize, baseLine = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label_text, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return image

def run_webcam(interpreter, labels, input_details, output_details, w, h, is_float):
    cap = cv2.VideoCapture(0)
    print("\n[INFO] Starting Webcam... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = detect_objects(frame, interpreter, labels, input_details, output_details, w, h, is_float)
        
        cv2.imshow('Drone Detector (Webcam)', frame)
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def run_image_folder(interpreter, labels, input_details, output_details, w, h, is_float):
    if not os.path.exists(IMAGE_OUTPUT_DIR): os.makedirs(IMAGE_OUTPUT_DIR)
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(IMAGE_INPUT_DIR, ext)))

    if not image_files:
        print(f"\n[ERROR] No images found in '{IMAGE_INPUT_DIR}' folder.")
        return

    print(f"\n[INFO] Processing {len(image_files)} images...")

    for img_path in image_files:
        filename = os.path.basename(img_path)
        image = cv2.imread(img_path)
        
        if image is None: continue

        # Run detection
        processed_image = detect_objects(image, interpreter, labels, input_details, output_details, w, h, is_float)

        # Save result
        save_path = os.path.join(IMAGE_OUTPUT_DIR, filename)
        cv2.imwrite(save_path, processed_image)
        print(f"  Saved: {save_path}")

    print(f"\n[SUCCESS] All images saved to '{IMAGE_OUTPUT_DIR}'")

# --- MAIN MENU ---
if __name__ == "__main__":
    # Load resources once
    interpreter, labels, input_details, output_details, w, h, is_float = load_model_and_labels()
    
    print("-" * 30)
    print(" DRONE DETECTION SYSTEM ")
    print("-" * 30)
    print("1. Use Live Webcam")
    print("2. Process Images from Folder")
    
    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == '1':
        run_webcam(interpreter, labels, input_details, output_details, w, h, is_float)
    elif choice == '2':
        run_image_folder(interpreter, labels, input_details, output_details, w, h, is_float)
    else:
        print("Invalid choice. Exiting.")