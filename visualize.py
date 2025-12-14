import cv2
import os

def visualize_yolo_bbox(image_path, label_path, class_names_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Get image dimensions
    img_height, img_width = image.shape[:2]

    # Load class names
    try:
        with open(class_names_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"Error: Could not load class names file {class_names_path}")
        return

    # Read the label file
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            # Parse the line: class_id, center_x, center_y, width, height
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            center_x, center_y, width, height = parts[1:]

            # Convert normalized coordinates to actual pixel values
            # Calculate top-left corner (x_min, y_min)
            x_center = int(center_x * img_width)
            y_center = int(center_y * img_height)
            box_width = int(width * img_width)
            box_height = int(height * img_height)

            x_min = int(x_center - box_width / 2)
            y_min = int(y_center - box_height / 2)
            x_max = int(x_center + box_width / 2)
            y_max = int(y_center + box_height / 2)

            # Draw the bounding box
            color = (0, 255, 0) # Green color for the box (OpenCV uses BGR)
            thickness = 2
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

            # Put the class label text
            if class_id < len(class_names):
                label = class_names[class_id]
                cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
            else:
                print(f"Warning: Class ID {class_id} out of range in class names file")

    else:
        print(f"Label file {label_path} not found.")

    # Display the image with boxes
    cv2.imshow("YOLO Bounding Box Visualization", image)
    cv2.waitKey(0) # Wait for a key press to close the window
    cv2.destroyAllWindows()

# Example Usage:
# Replace with your actual file paths
#image_file = "/Users/dhananjaydeshpande/Desktop/Columbia EE DES/Digital Signal Processing/Project/saffron_flower_dataset-master/dataset/train/images/6_jpg.rf.80b4de07b166ccbe7a838dec3d405f52.jpg"
#label_file = "/Users/dhananjaydeshpande/Desktop/Columbia EE DES/Digital Signal Processing/Project/saffron_flower_dataset-master/dataset/train/labels/6_jpg.rf.80b4de07b166ccbe7a838dec3d405f52.txt" # Corresponding .txt file in YOLO format
image_file = "/Users/dhananjaydeshpande/Desktop/Columbia EE DES/Digital Signal Processing/Project/saffron_flower_dataset-master/dataset/train/images/76_jpg.rf.e4dbfb263202595766cb80385f00e332.jpg"
label_file = "/Users/dhananjaydeshpande/Desktop/Columbia EE DES/Digital Signal Processing/Project/saffron_flower_dataset-master/dataset/train/labels/76_jpg.rf.e4dbfb263202595766cb80385f00e332.txt" # Corresponding .txt file in YOLO format

names_file = "/Users/dhananjaydeshpande/Desktop/Columbia EE DES/Digital Signal Processing/Project/saffron_flower_dataset-master/obj.names" # File containing class names, one per line

visualize_yolo_bbox(image_file, label_file, names_file)
