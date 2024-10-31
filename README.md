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

## Model Customization
The `src/testTrain` folder contains all the necessary files for training and testing your own model variations. You can modify different parameters and settings to experiment with the model's behavior and performance. This allows for customization and potential improvements in accuracy, processing speed, or specific use cases. Feel free to explore and adjust the training configurations according to your needs

## Detection Modes
- **Webcam**: Webcam for real-time detection
- **Video**: Video file for the model to recognize
- **Image**: Image for the model to recognize

## Technical
- **Computer Vision**: OpenCV
- **Machine Learning**: TensorFlow, MediaPipe
- **Backend**: Python, Eel
- **Frontend**: HTML/CSS/JavaScript
- **UI Runtime**: Chrome

## Contributing
Contributions are welcome! Feel free to submit pull requests with improvements, bug fixes, or new features. For discussions and suggestions, please open an issue.

## License
[MIT](https://choosealicense.com/licenses/mit/)

---
*Note: This application is for educational purposes and may require further refinement for production use.*