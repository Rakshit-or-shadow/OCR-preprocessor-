# Text Angle Detection

This project provides functionality to calculate the orientation of text in images. It includes methods for loading images, processing them, detecting text lines, calculating angles, and visualizing results. This can be particularly useful for applications in Optical Character Recognition (OCR) and document analysis.

## Features

- Load and process images to detect text orientation.
- Calculate in-plane and out-of-plane rotation angles.
- Visualize detected text orientation on images.
- Unit tests to ensure functionality and reliability.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/text-angle-detection.git
cd text-angle-detection
pip install -r requirements.txt
```

## Usage

To use the text angle detection functionality, you can run the `text_angle_detection.py` script. Here’s an example of how to call the main function:

```python
from src.text_angle_detection import calculate_text_orientation_to_camera

image_path = "path/to/your/image.jpg"
camera_matrix = None  # Replace with your camera matrix if available
dist_coeffs = None    # Replace with your distortion coefficients if available

result, message = calculate_text_orientation_to_camera(image_path, camera_matrix, dist_coeffs)
print(result)
```







(ignore from here)
## Running Tests 

To ensure that everything is working correctly, you can run the unit tests provided in the `tests` directory:

```bash
pytest tests/test_text_angle_detection.py
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
