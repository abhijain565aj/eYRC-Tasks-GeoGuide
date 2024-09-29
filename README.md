## eYRC 2023-24
# eYRC-Tasks-GeoGuide

- Team ID: GG_1380
- Author List: Abhi Jain, Sameer Arvind Patil, Anushka Singhal, Arya Suwalka

# Task-1A

This script implements a machine learning model for predicting employee retention based on provided features. It includes data preprocessing, feature encoding, model training, and validation steps using PyTorch. The final trained model is saved as a TorchScript file for deployment.

# Task-2A

In this task, the goal is to detect ArUco markers in an image and extract their center coordinates and orientations. The implementation detects ArUco markers and computes their corner coordinates and angles relative to the vertical axis.

# Task-2B

- **Task Overview**: The goal of this task is to classify various events such as 'combat', 'destroyed building', 'fire', 'humanitarian aid', and 'military vehicles' from input images using a deep learning model (ResNet18). This classification helps in recognizing key events from satellite imagery.

- **Model Training and Testing**: A ResNet18 model is fine-tuned with a custom classifier using a dataset of images representing different events. The trained model is then used to classify new images by loading the trained weights and returning the predicted event name.

- **Implementation**: The project includes scripts for training the model, classifying images during runtime, and processing event detection results for further analysis. The model is saved as event_classifier_model.pth and used in the main classification function (classify_event) to predict events from input images.

# Task-2C

- **Event Classification Using Pretrained ResNet from Task 2B**: This script classifies events in images using a modified ResNet-18 architecture, leveraging the model trained in Task 2B. The final classification layer has been customized to handle 5 event categories (combat, destroyed building, fire, humanitarian aid, and military vehicles).

- **Image Processing Pipeline**: The script extracts event images from the given arena image, applies transformations using OpenCV, and passes the images through the pretrained classifier from Task 2B. Detected events are saved and processed for further output.

# Task-2D

- **Task Overview**: The task involves reading latitude and longitude coordinates from a CSV file, simulating the tracking of paths using ArUco marker IDs, and validating the traversed paths through two test cases.

- **CSV Operations and Path Tracking**: The script reads a CSV file (lat_long.csv) to associate ArUco IDs with their respective latitude and longitude coordinates. It then simulates tracking two different paths (path1 and path2) by writing live coordinate data to a separate CSV (live_data.csv). Two test cases validate the correctness of the traversed paths.
