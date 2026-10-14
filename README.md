# SignEcho

**Real-Time Dynamic Sign Language Recognition for Mobile Communication**

SignEcho is a sign-language recognition project designed to help people communicate more easily in public places using mobile technology.

## Features

- Real-time hand landmark detection using MediaPipe.
- Dynamic sign recognition using a deep learning LSTM model.
- Recognition of hand movements from sequences of 100 frames.
- 63 hand-landmark features per frame.
- Text-to-speech output for recognized signs.
- Flutter mobile application for integrating sign recognition into a mobile interface.

## Technologies Used

- Python
- TensorFlow / Keras
- MediaPipe
- NumPy
- TensorFlow Lite
- Flutter and Dart

## Project Structure

```text
sign_language_project/
├── dynamic_dataset/
├── mobile_app/
├── model/
│   ├── dynamic_labels.pkl
│   ├── dynamic_sign_model.keras
│   └── dynamic_sign_model.tflite
├── dynamic_data_collection.py
├── dynamic_sign_test.py
├── dynamic_train_model.py
├── convert_to_tflite.py
└── README.md
```

## Model Details

- Sequence length: 100 frames
- Features per frame: 63
- Training dataset: 1,000 samples across 10 sign classes
- Reported test accuracy: 94.50%

## Current Sign Classes

Bye, emergency, hello, hungry, no, sorry, thankyou, toilets, where, yes.

## Running the Python Demo

1. Install the required Python dependencies.
2. Ensure the trained model files are present in the `model/` directory.
3. Run:

```bash
python dynamic_sign_test.py
```

Allow webcam access when prompted.

## Future Improvements

- Add more sign classes, including drink, phone, and sleep.
- Improve recognition accuracy for signs that are frequently confused.
- Complete and test mobile integration on Android and iOS.

## Project Status

The Python webcam recognition pipeline has been tested. Mobile integration and additional sign classes remain ongoing work.
