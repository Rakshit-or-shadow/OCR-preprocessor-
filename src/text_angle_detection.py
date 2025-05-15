#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 20:33:57 2025

@author: rohit.garg
"""

import cv2
import numpy as np
import math
from pytesseract import pytesseract

def calculate_text_orientation_to_camera(image_path, camera_matrix=None, dist_coeffs=None):
    """
    Calculate the orientation angle of text with respect to the camera.
    
    Args:
        image_path (str): Path to the input image
        camera_matrix (numpy.ndarray): 3x3 camera intrinsic matrix from calibration
        dist_coeffs (numpy.ndarray): Distortion coefficients from calibration
        
    Returns:
        dict: Orientation angles and confidence metrics
    """
    # Load and process image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Undistort image if calibration is available
    if camera_matrix is not None and dist_coeffs is not None:
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
        x, y, w, h = roi
        if all(v > 0 for v in [x, y, w, h]):
            image = image[y:y+h, x:x+w]
    
    # Convert to grayscale and preprocess
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use multiple preprocessing methods for better text region detection
    # Method 1: Adaptive thresholding
    thresh1 = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Method 2: Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Combine methods for robust text detection
    combined = cv2.bitwise_or(thresh1, edges)
    
    # Find text lines using probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(combined, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
    
    # If no lines found, try contour-based approach
    if lines is None or len(lines) == 0:
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
        
        if not filtered_contours:
            return None, "No text regions detected"
        
        # Use text regions to identify orientation
        text_boxes = []
        text_orientations = []
        
        for contour in filtered_contours:
            # Get minimum area rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            #box = np.int(box)
            
            # Order points in consistent manner
            ordered_box = order_points(box)
            text_boxes.append(ordered_box)
            
            # Calculate in-plane rotation
            width = np.linalg.norm(ordered_box[0] - ordered_box[1])
            height = np.linalg.norm(ordered_box[1] - ordered_box[2])
            
            # Determine orientation based on which dimension is larger
            angle = rect[2]
            if width < height:
                # Adjust angle for vertical text
                angle = angle - 90 if angle > -45 else angle
            
            # Normalize angle
            if angle < -45:
                angle = 90 + angle
            else:
                angle = -angle
                
            text_orientations.append(angle)
    else:
        # Extract lines and calculate their angles
        text_orientations = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:  # Vertical line
                angle = 90
            else:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Normalize to horizontal text baseline orientation
            if angle > 45 and angle <= 90:
                angle = angle - 90
            elif angle > 90 and angle <= 135:
                angle = angle - 180
            elif angle > -135 and angle <= -90:
                angle = angle + 180
            elif angle > -90 and angle <= -45:
                angle = angle + 90
                
            text_orientations.append(angle)
            
    
    # Calculate in-plane text orientation (average angle)
    in_plane_angle = np.median(text_orientations) if text_orientations else 0
    
    # For perspective analysis, use the largest text contour or combined contours
    if 'filtered_contours' in locals() and filtered_contours:
        # Find largest contour
        largest_idx = np.argmax([cv2.contourArea(cnt) for cnt in filtered_contours])
        largest_contour = filtered_contours[largest_idx]
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        #box = np.int(box)
        box = order_points(box)
    else:
        # Create a bounding box from detected lines
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.extend([(x1, y1), (x2, y2)])
            # Draw the line bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Display the result
            #cv2.imshow('Text Regions', image)
            #cv2.waitKey(0)
            #ßcv2.destroyAllWindows()
        
        if not points:
            return None, "Could not determine text orientation"
            
        points = np.array(points)
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        #box = np.int(box)
        box = order_points(box)
    
    # Calculate 3D orientation using camera information
    if camera_matrix is not None:
        # Use PnP solver to find text orientation in 3D
        # Assume standard text/document dimensions (adjust as needed)
        real_width = 210  # mm (A4 width)
        real_height = 297  # mm (A4 height)
        
        # Define 3D coordinates of text corners (z=0 plane)
        object_points = np.array([
            [0, 0, 0],                      # top-left
            [real_width, 0, 0],             # top-right
            [real_width, real_height, 0],   # bottom-right
            [0, real_height, 0]             # bottom-left
        ], dtype=np.float32)
        
        # Image points from detected text box
        image_points = np.array(box, dtype=np.float32)
        
        # Solve for pose
        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points, image_points, camera_matrix, 
            dist_coeffs if dist_coeffs is not None else np.zeros(5)
        )
        
        if success:
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Extract Euler angles
            # The three angles represent rotation around x, y, and z axes
            y_rot = math.atan2(rotation_matrix[2, 0], 
                              math.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
            x_rot = math.atan2(-rotation_matrix[2, 1], rotation_matrix[2, 2])
            z_rot = math.atan2(-rotation_matrix[1, 0], rotation_matrix[0, 0])
            
            # Convert to degrees
            pitch = math.degrees(x_rot)  # Rotation around X-axis (up/down tilt)
            yaw = math.degrees(y_rot)    # Rotation around Y-axis (left/right rotation)
            roll = math.degrees(z_rot)   # Rotation around Z-axis (in-plane rotation)
            
            # Normal vector of the text plane in camera coordinates
            # This vector points perpendicular to the text surface
            normal_vector = rotation_matrix[:, 2]  # Third column of rotation matrix
            
            # Calculate angle between text normal and camera optical axis ([0,0,1])
            camera_axis = np.array([0, 0, 1])
            dot_product = np.dot(normal_vector, camera_axis)
            normal_magnitude = np.linalg.norm(normal_vector)
            
            # Angle between text plane and camera's optical axis
            # 0° means text is perpendicular to camera, 90° means text is parallel to camera view
            text_camera_angle = np.degrees(np.arccos(dot_product / normal_magnitude))
            
            # Calculate obliqueness (how much the text is turned away from camera)
            # This is essentially the complement of the text_camera_angle
            obliqueness = 90 - text_camera_angle
            
            # Create result dictionary with comprehensive orientation information
            return {
                "in_plane_rotation": in_plane_angle,  # Traditional text angle in image plane
                "out_of_plane_orientation": {
                    "pitch": pitch,  # Up/down tilt
                    "yaw": yaw,      # Left/right rotation 
                    "roll": roll     # In-plane rotation
                },
                "text_camera_angle": text_camera_angle,  # Angle between text normal and camera axis
                "obliqueness": obliqueness,  # How obliquely the text is viewed (0=head-on)
                "normal_vector": normal_vector.tolist(),  # Normal vector of text plane
                "confidence": "high",
                "method": "camera_calibration"
            }, "Full 3D orientation analysis completed"
    
    # If no camera calibration available, use perspective estimation
    # Calculate vanishing points and perspective distortion
    # Measure the relative dimensions of text box
    width = np.linalg.norm(box[0] - box[1])
    height = np.linalg.norm(box[1] - box[2])
    
    # Calculate aspect ratio and distortion
    aspect_ratio = max(width, height) / min(width, height)
    distortion_factor = abs(aspect_ratio - 1)
    
    # Measure perspective distortion by comparing opposite sides
    top_side = np.linalg.norm(box[0] - box[1])
    bottom_side = np.linalg.norm(box[3] - box[2])
    left_side = np.linalg.norm(box[0] - box[3])
    right_side = np.linalg.norm(box[1] - box[2])
    
    # Calculate side ratios (values closer to 1 indicate less perspective distortion)
    horizontal_ratio = max(top_side, bottom_side) / min(top_side, bottom_side)
    vertical_ratio = max(left_side, right_side) / min(left_side, right_side)
    
    # Determine primary axis of distortion
    if horizontal_ratio > vertical_ratio:
        primary_axis = "horizontal"
        primary_distortion = horizontal_ratio
    else:
        primary_axis = "vertical"
        primary_distortion = vertical_ratio
    
    # Estimate the text-camera angle based on perspective distortion
    # More distortion suggests more oblique angle
    estimated_angle = min(85, (primary_distortion - 1) * 60)
    
    return {
        "in_plane_rotation": in_plane_angle,
        "estimated_text_camera_angle": estimated_angle,
        "perspective_analysis": {
            "distortion_factor": distortion_factor,
            "horizontal_ratio": horizontal_ratio,
            "vertical_ratio": vertical_ratio,
            "primary_distortion_axis": primary_axis
        },
        "confidence": "medium",
        "method": "perspective_estimation"
    }, "Perspective-based orientation analysis completed"

def order_points(pts):
    """
    Order points in clockwise order starting from top-left
    """
    # Sort by sum of coordinates (x+y), the lowest is top-left, highest is bottom-right
    s = pts.sum(axis=1)
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    
    # Sort by difference of coordinates (x-y), the lowest is top-right, highest is bottom-left
    diff = np.diff(pts, axis=1)
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
    
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

"""
def order_points(pts):
    
    Order points in clockwise order starting from top-left using cv2.minAreaRect.
    
    # Get the minimum area rectangle
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)  # Get the 4 corner points
    box = np.array(box, dtype="float32")

    # Sort the points in clockwise order starting from the top-left
    # Top-left will have the smallest sum, bottom-right the largest sum
    s = box.sum(axis=1)
    top_left = box[np.argmin(s)]
    bottom_right = box[np.argmax(s)]

    # Top-right will have the smallest difference, bottom-left the largest difference
    diff = np.diff(box, axis=1)
    top_right = box[np.argmin(diff)]
    bottom_left = box[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
"""

def visualize_orientation(image_path, result, output_path=None):
    """
    Visualize the detected text orientation on the image
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Create a copy for visualization
    vis_image = image.copy()
    
    # Draw text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    
    # Add in-plane rotation
    cv2.putText(vis_image, f"In-plane: {result['in_plane_rotation']:.1f} deg", 
               (20, 30), font, font_scale, (0, 0, 255), font_thickness)
    
    if 'estimated_text_camera_angle' in result:
        # For perspective estimation method
        cv2.putText(vis_image, f"Camera angle: {result['estimated_text_camera_angle']:.1f} deg", 
                   (20, 60), font, font_scale, (0, 0, 255), font_thickness)
    elif 'text_camera_angle' in result:
        # For camera calibration method
        cv2.putText(vis_image, f"Camera angle: {result['text_camera_angle']:.1f} deg", 
                   (20, 60), font, font_scale, (0, 0, 255), font_thickness)
        cv2.putText(vis_image, f"Pitch: {result['out_of_plane_orientation']['pitch']:.1f} deg", 
                   (20, 90), font, font_scale, (0, 0, 255), font_thickness)
        cv2.putText(vis_image, f"Yaw: {result['out_of_plane_orientation']['yaw']:.1f} deg", 
                   (20, 120), font, font_scale, (0, 0, 255), font_thickness)
    
    # Save or show the result
    if output_path:
        cv2.imwrite(output_path, vis_image)
        
    # Display the result
    #cv2.imshow('Text Regions', vis_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return vis_image

def rotate_image_to_upright(image_path, angle, output_path=None):
    """
    Rotate the image by the given angle (in degrees) to deskew text.
    Positive angle rotates counterclockwise.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    # Negative angle to deskew (make text vertical)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if output_path:
        cv2.imwrite(output_path, rotated)
    return rotated

"""
def draw_text_boxes(image_path, output_path="rotated_with_boxes.jpg"):
    image = cv2.imread(image_path)
    boxes = pytesseract.image_to_boxes(image)
    h, w = image.shape[:2]
    for b in boxes.splitlines():
        b = b.split(' ')
        x1, y1, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        # Tesseract's y origin is at the bottom, OpenCV's at the top
        cv2.rectangle(image, (x1, h - y2), (x2, h - y1), (0, 255, 0), 2)
    cv2.imwrite(output_path, image)
    print(f"Bounding boxes drawn and saved to {output_path}")
"""

# Example usage
def main():
    # Example camera matrix (replace with your calibration results)
    camera_matrix = np.array([
        [1000, 0, 320],  # fx, 0, cx
        [0, 1000, 240],  # 0, fy, cy
        [0, 0, 1]        # 0, 0, 1
    ])
    dist_coeffs = np.zeros((5, 1))  # Distortion coefficients
    
    image_path = r"C:\Users\raksh\OneDrive\Documents\ocr\ocr-angle-preprocessor\test-file\WhatsApp Image 2025-05-13 at 12.45.10 PM.jpeg"
    
    try:
        # Load the image to determine its dimensions
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        (h, w) = image.shape[:2]
        
        # Determine initial orientation based on aspect ratio
        if w > h:
            initial_orientation = 180  # Landscape mode
        else:
            initial_orientation = 90  # Portrait mode (assume upright portrait)

        # Calculate orientation with calibration
        result, message = calculate_text_orientation_to_camera(
            image_path, camera_matrix, dist_coeffs
        )
        result, message = calculate_text_orientation_to_camera(
            image_path, None, None
        )
        
        if result:
            print(f"Text Orientation Analysis ({result['method']}):")
            print(f"In-plane rotation: {result['in_plane_rotation']:.2f} degrees")

            # Rotate image to make text upright
            adjusted_angle = -result['in_plane_rotation']  # Flip the sign of the in-plane angle
            rotated_img = rotate_image_to_upright(image_path, adjusted_angle, "rotated_upright.jpg")
            print("Rotated image saved as 'rotated_upright.jpg'")
            
            # Calculate the final orientation
            final_orientation = initial_orientation + adjusted_angle
            final_orientation = final_orientation % 360  # Normalize to [0, 360)
            print(f"Final orientation of the image: {final_orientation:.2f} degrees")
                            
            if result['method'] == 'camera_calibration':
                print("\nFull 3D Orientation:")
                print(f"Text-Camera Angle: {result['text_camera_angle']:.2f} degrees")
                print(f"Obliqueness: {result['obliqueness']:.2f} degrees")
                print(f"Pitch: {result['out_of_plane_orientation']['pitch']:.2f} degrees")
                print(f"Yaw: {result['out_of_plane_orientation']['yaw']:.2f} degrees")
                print(f"Roll: {result['out_of_plane_orientation']['roll']:.2f} degrees")
                print(f"Text plane normal vector: {result['normal_vector']}")
            else:
                print(f"\nEstimated Text-Camera Angle: {result['estimated_text_camera_angle']:.2f} degrees")
                print("\nPerspective Analysis:")
                print(f"Distortion factor: {result['perspective_analysis']['distortion_factor']:.2f}")
                print(f"Horizontal ratio: {result['perspective_analysis']['horizontal_ratio']:.2f}")
                print(f"Vertical ratio: {result['perspective_analysis']['vertical_ratio']:.2f}")
                print(f"Primary distortion axis: {result['perspective_analysis']['primary_distortion_axis']}")
            
            # Visualize results
            output_image = visualize_orientation(image_path, result, "orientation_result.jpg")
            if output_image is not None:
                print("\nVisualization saved to 'orientation_result.jpg'")
        else:
            print(f"Error: {message}")
            
    except Exception as e:
        print(f"Exception occurred: {str(e)}")

if __name__ == "__main__":
    main()
""""
import os

def main():
    # Example camera matrix (replace with your calibration results)
    camera_matrix = np.array([
        [1000, 0, 320],  # fx, 0, cx
        [0, 1000, 240],  # 0, fy, cy
        [0, 0, 1]        # 0, 0, 1
    ])
    dist_coeffs = np.zeros((5, 1))  # Distortion coefficients
    
    input_folder = r"c:\Users\raksh\OneDrive\Documents\ocr\text-angle-detection\tests"  # Folder containing images
    output_folder = r"c:\Users\raksh\OneDrive\Documents\ocr\text-angle-detection\output"  # Folder to save results

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpeg"):
            image_path = os.path.join(input_folder, filename)
            print(f"Processing: {image_path}")
            
            try:
                # Calculate orientation with calibration
                result, message = calculate_text_orientation_to_camera(
                    image_path, camera_matrix, dist_coeffs
                )
                
                if result:
                    print(f"Text Orientation Analysis ({result['method']}):")
                    print(f"In-plane rotation: {result['in_plane_rotation']:.2f} degrees")

                    # Rotate image to make text upright
                    adjusted_angle = -result['in_plane_rotation']  # Flip the sign of the in-plane angle
                    output_image_path = os.path.join(output_folder, f"rotated_{filename}")
                    rotated_img = rotate_image_to_upright(image_path, adjusted_angle, output_image_path)
                    print(f"Rotated image saved as '{output_image_path}'")
                    
                    # Visualize results
                    visualization_path = os.path.join(output_folder, f"visualization_{filename}")
                    output_image = visualize_orientation(image_path, result, visualization_path)
                    if output_image is not None:
                        print(f"Visualization saved to '{visualization_path}'")
                else:
                    print(f"Error processing {filename}: {message}")
                    
            except Exception as e:
                print(f"Exception occurred while processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()

"""