from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
from skimage.measure import regionprops
import cv2

my_new_model = YOLO("results/500_epochs-2/weights/best.pt")

new_image = 'test.jpg'
new_results = my_new_model.predict(new_image, conf=0.5)  #Adjust conf threshold

new_result_array = new_results[0].plot()
plt.figure(figsize=(12, 12))
plt.imshow(new_result_array)
plt.savefig("result.jpg")

new_result = new_results[0]
extracted_masks = new_result.masks.data
print(extracted_masks.shape)

masks_array = extracted_masks.cpu().numpy()
plt.imshow(masks_array[0])
plt.savefig("result_mask.jpg")

class_names = new_result.names.values()

# Extract the boxes, which likely contain class IDs
detected_boxes = new_result.boxes.data
# Extract class IDs from the detected boxes
class_labels = detected_boxes[:, -1].int().tolist()
# Initialize a dictionary to hold masks by class
masks_by_class = {name: [] for name in new_result.names.values()}

# Iterate through the masks and class labels
for mask, class_id in zip(extracted_masks, class_labels):
    class_name = new_result.names[class_id]  # Map class ID to class name
    masks_by_class[class_name].append(mask.cpu().numpy())


for class_name, masks in masks_by_class.items():
    print(f"Class Name: {class_name}, Number of Masks: {len(masks)}")

crack_masks = masks_by_class['crack']
orig_img = new_result.orig_img


resized_mask = cv2.resize(crack_masks[0], (640, 640))
normalized_mask = resized_mask / resized_mask.max()


plt.imshow(orig_img, cmap='gray')  # Display the original image
plt.imshow(normalized_mask, cmap='jet', alpha=0.3)  # Overlay the mask with transparency
plt.axis('off')  # Turn off axis labels
plt.savefig("result_mask1.jpg")
#plt.show()


props_list = []


for class_name, masks in masks_by_class.items():
    
    for mask in masks:
        # Convert the mask to an integer type if it's not already
        mask = mask.astype(int)
        # Apply regionprops to the mask
        props = regionprops(mask)
        for prop in props:
            # Extract region properties
            area = prop.area
            perimeter = prop.perimeter
            major_axis_length = prop.major_axis_length
            minor_axis_length = prop.minor_axis_length
            centroid = prop.centroid
            bbox = prop.bbox
            eccentricity = prop.eccentricity
            solidity = prop.solidity
            convex_area = prop.convex_area

            # Calculate aspect ratio
            aspect_ratio = major_axis_length / minor_axis_length if minor_axis_length > 0 else 0

            # Bounding box aspect ratio
            bbox_width = bbox[3] - bbox[1]  # max_col - min_col
            bbox_height = bbox[2] - bbox[0]  # max_row - min_row
            bbox_aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0

            # Calculate circularity
            circularity = (4 * 3.141592653589793 * area) / (perimeter ** 2) if perimeter > 0 else 0

            # Append properties to the list
            props_list.append({
                'Class Name': class_name,
                'Area': area,
                'Perimeter': perimeter,
                'Major Axis Length': major_axis_length,
                'Minor Axis Length': minor_axis_length,
                'Aspect Ratio': aspect_ratio,
                'Bounding Box Aspect Ratio': bbox_aspect_ratio,
                'Circularity': circularity,
                'Centroid': centroid,
                'Bounding Box': bbox,
                'Eccentricity': eccentricity,
                'Solidity': solidity,
                'Convex Area': convex_area
            })

# Convert the list of dictionaries to a DataFrame
props_df = pd.DataFrame(props_list)

# Save the DataFrame to a CSV file
props_df.to_csv('extracted_details.csv', index=False)

#print(props_df)

