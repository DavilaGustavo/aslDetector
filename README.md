<p align="center">
  <img src="https://github.com/user-attachments/assets/b8e91a75-371d-4f85-a8a4-cdcc0c28fa97" alt="animated gif">
</p>

## Description
ASL Detector is a computer program that recognizes American Sign Language (ASL) in real time. The program can work with your webcam, videos, or pictures. Using machine learning technology, it can detect and understand hand signs, showing you the results right away on your screen.

The program uses a machine learning model to recognize hand signs from the ASL alphabet. With tools like TensorFlow and MediaPipe, the system can track hands and understand signs with high accuracy.

## Features
- 24 ASL letters recognition (A-Z, except J & Z)
  - Capable of doing it in webcam, videos and images
- Intuitive interface
- Instant visual feedback

## How to use

### Prerequisites
- Python 3.8+
- Google Chrome

### Setup
1. Clone the repository:
```bash
git clone https://github.com/DavilaGustavo/aslDetector.git
cd aslDetector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/635ceb77-a603-4be0-80df-ee0906990922">
</p>

## Detection Modes
- **Webcam**: Webcam for real-time detection
- **Video**: Video file for the model to recognize
- **Image**: Image for the model to recognize

## Model Customization
The `src/testTrain` folder contains all the necessary files for training and testing your own model variations. You can modify different parameters and settings to experiment with the model's behavior and performance. This allows for customization and potential improvements in accuracy, processing speed, or specific use cases. Feel free to explore and adjust the training configurations according to your needs.

It includes complete source code for the model creation pipeline. You'll find scripts for capturing and saving images for the dataset, converting these images to a .csv dataset format, and both training and testing the model. While the image files themselves are not included due to size constraints, all other necessary files are present. This allows you to either start from scratch with your own images or skip certain steps by using the provided intermediate files.

<div align="center">
    <img src="https://github.com/user-attachments/assets/fe3b644a-85f6-41d2-83f2-9d87754c9d3b" width="30%" />
    &nbsp;&nbsp;
    <img src="https://github.com/user-attachments/assets/7e1198db-569b-44e6-8bba-4af182f6764e" width="30%" />
    &nbsp;&nbsp;
    <img src="https://github.com/user-attachments/assets/f7dc9f5b-632a-4483-b19b-7440a8babdb3" width="30%" />
</div>

## Technical
- **Computer Vision**: OpenCV
- **Machine Learning**: TensorFlow, MediaPipe
- **Backend**: Python, Eel
- **Frontend**: HTML/CSS/JavaScript
- **UI Runtime**: Chrome

## Datasets used
- [grassknoted/ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- [debashishsau/ASL(American Sign Language) Alphabet Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset/)
- [jordiviader/American Sign Language Alphabet (Static)](https://www.kaggle.com/datasets/jordiviader/american-sign-language-alphabet-static)
- [lexset/Synthetic ASL Alphabet](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet)
- Around 19k images taken from the webcam with `src/testTrain/cameraToData`.

## Contributing
Contributions are welcome! Feel free to submit pull requests with improvements, bug fixes, or new features. For discussions and suggestions, please open an issue.

## License
[MIT](https://choosealicense.com/licenses/mit/)

---
*Note: This application is for educational purposes and may require further refinement for production use.*