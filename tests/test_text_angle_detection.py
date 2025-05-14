import unittest
import numpy as np
import cv2
from src.text_angle_detection import calculate_text_orientation_to_camera, rotate_image_to_upright, visualize_orientation

class TestTextAngleDetection(unittest.TestCase):

    def setUp(self):
        self.image_path = "test_image.jpg"  # Path to a test image
        self.camera_matrix = np.array([
            [1000, 0, 320],
            [0, 1000, 240],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.zeros((5, 1))

    def test_calculate_text_orientation_to_camera(self):
        result, message = calculate_text_orientation_to_camera(self.image_path, self.camera_matrix, self.dist_coeffs)
        self.assertIsNotNone(result)
        self.assertIn("in_plane_rotation", result)
        self.assertIn("method", result)

    def test_rotate_image_to_upright(self):
        angle = 45  # Example angle
        rotated_image = rotate_image_to_upright(self.image_path, angle)
        self.assertIsNotNone(rotated_image)
        self.assertEqual(rotated_image.shape[0], cv2.imread(self.image_path).shape[0])  # Check height remains the same

    def test_visualize_orientation(self):
        result = {
            "in_plane_rotation": 30,
            "text_camera_angle": 45,
            "out_of_plane_orientation": {
                "pitch": 10,
                "yaw": 20,
                "roll": 30
            }
        }
        output_image = visualize_orientation(self.image_path, result, "output_image.jpg")
        self.assertIsNotNone(output_image)

if __name__ == "__main__":
    unittest.main()