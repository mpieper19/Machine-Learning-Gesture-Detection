# Gesture Detection with Machine Learning

A Python-based project for gesture recognition using machine learning.

## Table of Contents

- [About the Project](#about-the-project)
- [Built With](#built-with)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Collection](#data-collection)
  - [Dataset Creation](#dataset-creation)
  - [Model Training](#model-training)
  - [Gesture Detection](#gesture-detection)
- [Credits](#credits)

## About the Project

This project implements a gesture recognition system using Python and machine learning techniques. It collects data, preprocesses it, trains a model, and detects gestures in real-time. It serves as a set of prerequisite scripts for future projects I plan on working on.

## Built With

- Python
- OpenCV
- Numpy
- Pandas
- Scikit Learn

Use `pip install -r requirements.txt` to install the required dependencies.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/mpieper19/Machine-Learning-Gesture-Detection.git
   cd Machine-Learning-Gesture-Detection
   ```

2. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

The project comprises four main scripts:

### Data Collection

**Script**: `1_data_collector.py`  
*Collects images for different hand gestures.*

- Running the script will first create a `captured_frames` folder in this repo's directory.
- It will then prompt you with a video capture, using your first registered capture device.
- This script allows you to save/create up to 9 gestures.
  - Pressing any key `0-9` will create a corresponding folder in the `captured_frames` folder to save images to.
- After creating a folder/gesture, pressing `c` initiates a 30-second timer, during which frames are continuously saved.

### Dataset Creation

**Script**: `2_dataset_creator.py`  
*Processes the collected data and organizes it into a dataset suitable for training.*

- This creates a csv file called `gesture_data.csv` in this repo's directory.
- This script analyses each image, from each labeled folder (`0-9`), and detects present hand landmark data (21 points).
- The dataset stores the `x` and `y` coordinates of each landmark point.
  - This results in 42 features per gesture.
- The dataset's labels correspond to the gesture's labeled folder.

### Model Training

**Script**: `3_train_model.py`  
*Trains a machine learning model using the prepared dataset.*

- Using the "Scikit-learn" library, the chosen model to train the dataset is the "Gradient Boosting Classifier" model.
- This script uses the saved `gesture_data.csv` file and saves the trained model as `gesture_model.pkl`.

### Gesture Detection

**Script**: `4_gesture_detector.py`  
*Uses the trained model to detect and recognize hand gestures in real-time.*

- This script will prompt you with a video capture, using your first registered capture device.
- Provided that the `labels_dict` code-line is edited accordingly, you can see predicted gestures in real time.

## Credits

- **Computer Vision Engineer**: This project was inspired by his tutorial on sign recognition. - [Youtube Tutorial](https://youtu.be/MJCSjXepaAM?si=4izT7LKkjPhpcodP), [GitHub](https://github.com/computervisioneng).
- **Vinu Balan**: Helped me understand how to store landmark data into a csv file. - [GitHub Repo link](https://github.com/Vinu-Balan/LazzyPro-PC-Control-Using-Hand-Gesture/blob/main/landmarks_medium.csv).
- **Google MediaPipe**: Helped me understand landmark detection, and overall understanding of the mediapipe package. - [Documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker), [More Documentation](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html)

## License

This project is licensed under the MIT License.
