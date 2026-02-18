# Attention Monitor 🎯

Real-time attention and concentration monitoring system using advanced computer vision techniques.

## Features

- 🎥 **Real-time Person Detection** - YOLOv8-based detection and tracking
- 👁️ **Pose Estimation** - MediaPipe for head position and landmark analysis
- 📊 **Concentration Scoring** - Dynamic scoring based on posture and behavior
- 📱 **Distraction Detection** - Phone and distraction identification
- 👥 **Multi-person Tracking** - Track multiple people simultaneously
- 📈 **Metrics & History** - Concentration history and averaging

## How It Works

### Concentration Metrics

The system calculates a concentration score (0-100) based on:

- **Head Down (70% weight)** → Good concentration ✓
- **Head Turned/Raised (30% weight)** → Low concentration ✗
- **Phone Detected (90% penalty)** → Major distraction ⚠️

### Person Tracking

Each detected person is assigned a unique ID and their metrics are tracked over time:
- Continuous monitoring of concentration levels
- 30-frame history for averaging
- Real-time feedback and alerts

## Tech Stack

- **YOLOv8** (`yolov8n.pt`) - Person detection and bounding boxes
- **MediaPipe** - Pose estimation and facial landmarks
- **OpenCV** - Video capture and processing
- **NumPy** - Numerical computations and data handling

## Installation

### Prerequisites
- Python 3.8+
- Webcam (for real-time monitoring)

### Setup

```bash
# Clone the repository
git clone https://github.com/Omar-Bou01/attention_test.git
cd attention_test

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
opencv-python>=4.8.0
mediapipe>=0.10.0
ultralytics>=8.0.0
numpy>=1.24.0
```

## Usage

```bash
python attention_test.py
```

### Controls

- **ESC** - Exit the application
- **q** - Quit

## Output

The application displays:
- Bounding boxes around detected persons
- Head pose indicators
- Real-time concentration score per person
- Phone detection warnings
- Concentration history graph

## Project Structure

```
attention_test.py      # Main application
yolov8n.pt            # YOLOv8 nano model weights
.gitignore            # Git ignore configuration
README.md             # This file
```

## Performance Notes

- Uses YOLOv8 nano model for faster inference
- Optimized for real-time processing
- Tested on standard webcams
- Recommended: GPU acceleration for better performance

## Future Enhancements

- [ ] Eye contact detection
- [ ] Fatigue detection
- [ ] Body language analysis
- [ ] Data export and analytics
- [ ] Web dashboard integration
- [ ] Multi-camera support

## Author

Omar Bou - [GitHub](https://github.com/Omar-Bou01)

## License

MIT License - feel free to use and modify

---

**Made with ❤️ for attention monitoring**
