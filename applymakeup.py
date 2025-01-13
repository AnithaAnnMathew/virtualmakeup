import os
import json
import cv2
import numpy as np

# Define facial regions
FACIAL_REGIONS = {
    "face_shape_right": list(range(0, 11)),
    "inner_lower_lips": list(range(11, 26)),
    "left_eye upper": list(range(26, 39)),
    "left_eye lower": list(range(39, 48)),
    "right_eye upper": list(range(48, 60)),
    "right_eye lower": list(range(60, 70)),
    "left_eyebrow upper": list(range(70, 82)),
    "left_eyebrow lower": list(range(82, 92)),
    "right_eyebrow upper": list(range(92, 103)),
    "right_eyebrow lower": list(range(103, 114)),
    "face_shape_left": list(range(114, 135)),
    "nose": list(range(135, 152)),
    "outer_upper_lips": list(range(152, 165)),
    "outer_lower_lips": list(range(165, 180)),
    "inner_upper_lips": list(range(180, 194)),

}

# Exclude jawline points (21, 32, 43, 52, ..., 109)
EXCLUDE_POINTS = list(range(21, 110, 11))

# Extract regions
def extract_regions(data, regions):
    shapes = data['shapes']
    for shape in shapes:
        label = int(shape['label'])
        x, y = shape['points'][0]
        if label in EXCLUDE_POINTS:
            continue
        for region, indices in FACIAL_REGIONS.items():
            if label in indices:
                new_point = {"label": label, "x": x, "y": y}
                regions[region].append(new_point)
    return regions

# Apply eyeshadow
def apply_eyeshadow(upper_eyelid_points, lower_eyebrow_points, image, color, intensity):
    eyeline_inside = upper_eyelid_points[0]
    eyeline_middle = upper_eyelid_points[len(upper_eyelid_points) // 2]
    eyeline_outside = upper_eyelid_points[-1]
    eyebrow_inside = lower_eyebrow_points[-1]
    eyebrow_middle = lower_eyebrow_points[len(upper_eyelid_points) // 2]
    eyebrow_outside = lower_eyebrow_points[0]

    #Define midpoints between eye and eyebrows for approximate area of application
    mid_inside_x = (eyeline_inside['x'] + eyebrow_inside['x']) / 2
    mid_middle_x = (eyeline_middle['x'] + eyebrow_middle['x']) / 2
    mid_outside_x = (eyeline_outside['x'] + eyebrow_outside['x']) / 2

    mid_inside_y = (eyeline_inside['y'] + eyebrow_inside['y']) / 2
    mid_middle_y = (eyeline_middle['y'] + eyebrow_middle['y']) / 2
    mid_outside_y = (eyeline_outside['y'] + eyebrow_outside['y']) / 2

    mp = [(mid_outside_x, mid_outside_y), (mid_middle_x, mid_middle_y), (mid_inside_x, mid_inside_y) ]

    mask = np.zeros_like(image)
    polygon = np.array(
        [(int(p['x']), int(p['y'])) for p in upper_eyelid_points] + mp, np.int32
    )
    cv2.fillPoly(mask, [polygon], color)
    blurred_eyeshadow = cv2.GaussianBlur(mask, (15, 15), 0)
    return cv2.addWeighted(image, 1, blurred_eyeshadow, intensity, 0)
    #return cv2.polylines(image_out, [polygon], isClosed=True, color=(0, 255, 0), thickness=1)
    

# Apply lipstick
def apply_lipstick(outer_lips, inner_lips, image, color, intensity):
    mask = np.zeros_like(image)
    polygon = np.array(
        [(int(p['x']), int(p['y'])) for p in outer_lips] +
        [(int(p['x']), int(p['y'])) for p in inner_lips],
        np.int32
    )
    cv2.fillPoly(mask, [polygon], color)
    blurred_lipstick = cv2.GaussianBlur(mask, (15, 15), 0)
    return cv2.addWeighted(image, 1, blurred_lipstick, intensity, 0)
    #return cv2.polylines(image_out, [polygon], isClosed=True, color=(0, 255, 0), thickness=1)



# Apply eyeliner
def apply_eyeliner(upper_eyelid_points, image, color, thickness):
    for i in range(len(upper_eyelid_points) - 1):
        start_point = (int(upper_eyelid_points[i]['x']), int(upper_eyelid_points[i]['y']))
        end_point = (int(upper_eyelid_points[i + 1]['x']), int(upper_eyelid_points[i + 1]['y']))
        cv2.line(image, start_point, end_point, color, thickness)
    return image


        
# Main function
def main():
    # File paths
    json_file = '000506.json'
    image_file = '000506.png'

    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Read the image
    image = cv2.imread(image_file)
    cv2.imshow("original Image", image)
    cv2.waitKey(0)
    h, w, _ = image.shape
    shapes = data['shapes']
    
    regions = {region: [] for region in FACIAL_REGIONS}
    regions = extract_regions(data, regions)

    # Apply eyeshadow
    image = apply_eyeshadow(regions['left_eye upper'], regions['left_eyebrow lower'], image, (128, 0, 128), 0.5)
    image = apply_eyeshadow(regions['right_eye upper'], regions['right_eyebrow lower'], image, (128, 0, 128), 0.5)
    

    # Apply lipstick
    image = apply_lipstick(regions['inner_lower_lips'], regions['outer_lower_lips'], image, (0, 0, 255), 0.7)
    image = apply_lipstick(regions['outer_upper_lips'], regions['inner_upper_lips'], image, (0, 0, 255), 0.7)

    
    # # Apply eyeliner
    image = apply_eyeliner(regions['left_eye upper'], image, (0, 0, 0), 1)
    image = apply_eyeliner(regions['right_eye upper'], image, (0, 0, 0), 1)
    
    # Display output
    cv2.imshow("Makeup Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
