# 🇧🇩 Bangladeshi Taka Detection Using YOLOv8

A computer vision application that detects Bangladeshi Taka banknotes from uploaded images using a trained **YOLO (You Only Look Once) object detection model** and exposes the detection functionality through a **FastAPI REST API**.

The application accepts an image, performs object detection, identifies the detected banknote classes, and returns the results in JSON format.

## 📌 Project Overview

Bangladeshi Taka Detection is a deep learning-based computer vision project designed to identify Bangladeshi currency notes from images. It uses a trained YOLO model (`best.pt`) to detect banknotes and FastAPI to provide an API endpoint for image-based inference.

The project can serve as a foundation for currency recognition applications, assistive technologies, and real-time computer vision systems.

### Key Features

- **YOLO-based object detection:** Uses a trained YOLO model to detect Bangladeshi Taka banknotes.
- **FastAPI REST API:** Provides an HTTP endpoint for image uploads and detection.
- **Image upload support:** Accepts images through a multipart form-data request.
- **Confidence threshold:** Uses a confidence threshold of `0.25` for inference.
- **Class identification:** Returns the detected banknote class names.
- **Duplicate removal:** Removes repeated class names from the response.
- **Automatic file cleanup:** Deletes uploaded image files after processing.
- **JSON responses:** Returns structured detection results for easy integration with other applications.
- **Interactive API documentation:** Supports automatic Swagger UI documentation through FastAPI.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLO / Ultralytics | Banknote object detection |
| FastAPI | REST API development |
| Uvicorn | ASGI server |
| Pillow (PIL) | Image loading and processing |
| UUID | Unique uploaded image filenames |
| Python OS module | File and directory management |

## 📂 Project Structure

```text
bangladeshi-taka-detection/
│
├── main1.py                 # FastAPI application and detection endpoint
├── best.pt                 # Trained YOLO model weights
├── requirements.txt        # Python dependencies
├── .gitignore              # Files excluded from Git
├── README.md               # Project documentation
│
└── uploads/                # Temporary uploaded images
```

**Note:** The `uploads/` directory is created automatically when the application starts. Uploaded images are deleted after processing. The trained `best.pt` file must be available at the configured model path.

## ⚙️ Installation and Setup

### Prerequisites

Install the following before running the project:

- Python 3.10 or a compatible Python version supported by the selected dependencies.
- pip (Python package manager).
- A trained YOLO model file named `best.pt`.

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/bangladeshi-taka-detection.git
```

Navigate to the project directory:

```bash
cd bangladeshi-taka-detection
```

Replace `YOUR_USERNAME` and the repository name with the actual GitHub details.

### Step 2: Create a Virtual Environment

On Windows:

```bash
python -m venv .venv
```

Activate the environment:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

Create a `requirements.txt` file containing:

```text
fastapi
uvicorn[standard]
ultralytics
pillow
python-multipart
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

For reproducible deployments, pin compatible dependency versions after testing the application in your environment.

### Step 4: Add the Trained Model

Place the trained YOLO model file in the project root directory:

```text
bangladeshi-taka-detection/
├── main1.py
└── best.pt
```

The application loads the model using:

```python
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)
```

**Important:** The `best.pt` file contains the trained model weights and is required to run inference. The model architecture and class labels depend on the model used during training.

### Step 5: Run the API Server

Start the FastAPI application with Uvicorn:

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

The `--reload` option is useful during development. For production, run without `--reload` and configure appropriate server resources.

## 🚀 API Endpoints

### 1. Root Endpoint

**Endpoint:** `GET /`

Checks whether the API is running.

Example request:

```http
GET /
```

Example response:

```json
{
  "message": "Bangladeshi Taka Detection API is running.",
  "endpoint": "/detect"
}
```

### 2. Banknote Detection Endpoint

**Endpoint:** `POST /detect`

Uploads an image and detects the Bangladeshi Taka banknote classes recognized by the trained model.

**Request format:** `multipart/form-data`

| Parameter | Type | Description |
|---|---|---|
| `file` | Image file | The image to analyze |

The uploaded image must be supplied using the form field named `file`.

#### Example: Successful Detection

```json
{
  "result": "DETECTED",
  "notes": [
    "100_taka",
    "500_taka"
  ]
}
```

The class names shown are illustrative. Actual names depend on the labels configured in the trained YOLO model.

#### Example: No Banknotes Detected

```json
{
  "result": "NONE",
  "notes": []
}
```

This response indicates that the model did not return any detections above the configured confidence threshold.

## 🧪 How to Test the API

### Option 1: Swagger UI

FastAPI provides interactive API documentation.

1. Start the application.
2. Open `http://127.0.0.1:8000/docs` in your browser.
3. Expand the `POST /detect` endpoint.
4. Click **Try it out**.
5. Upload an image containing one or more Bangladeshi Taka banknotes.
6. Click **Execute**.
7. Review the HTTP status code and JSON response.

### Option 2: cURL

Run the following command from a terminal:

```bash
curl -X POST "http://127.0.0.1:8000/detect" \
  -F "file=@sample_taka.jpg"
```

Replace `sample_taka.jpg` with the path to an existing image.

On Windows Command Prompt, the equivalent command is:

```cmd
curl -X POST "http://127.0.0.1:8000/detect" -F "file=@sample_taka.jpg"
```

### Option 3: Python Requests

Install the Requests library if it is not already installed:

```bash
pip install requests
```

Example client script:

```python
import requests

url = "http://127.0.0.1:8000/detect"

with open("sample_taka.jpg", "rb") as image_file:
    response = requests.post(
        url,
        files={"file": image_file}
    )

print("Status Code:", response.status_code)
print("Response:", response.json())
```

## 🔍 How the System Works

The application follows these processing steps:

1. **Receive image:** The API accepts an image through the `/detect` endpoint.
2. **Generate a unique filename:** A UUID is used to prevent filename collisions.
3. **Save the image temporarily:** The uploaded file is written to the `uploads/` directory.
4. **Preprocess the image:** Pillow opens the image and converts it to RGB format.
5. **Run YOLO inference:** The trained model predicts the objects using a confidence threshold of `0.25`.
6. **Extract predictions:** The application reads each detection's class ID and maps it to the corresponding class name.
7. **Remove duplicate class names:** Repeated class names are removed from the response.
8. **Return JSON:** The API returns `DETECTED` with the recognized classes or `NONE` if no classes are detected.
9. **Delete the temporary file:** The uploaded image is removed after processing, including when inference raises an exception after the file has been saved.

### System Workflow

```text
User
  |
  v
Upload Banknote Image
  |
  v
FastAPI /detect Endpoint
  |
  v
Save Image Temporarily
  |
  v
Convert Image to RGB
  |
  v
Trained YOLO Model (best.pt)
  |
  v
Extract Detected Class Names
  |
  v
Remove Duplicate Class Names
  |
  v
Return JSON Response
  |
  v
Delete Temporary Image
```

## 🧠 Model Information

The application uses a trained YOLO model loaded through the Ultralytics Python library.

| Parameter | Configuration |
|---|---|
| Model weights | `best.pt` |
| Framework | Ultralytics YOLO |
| Task | Object detection |
| Input | Uploaded image |
| Confidence threshold | `0.25` |
| Output | Detected class names |
| Inference method | `model.predict()` |

**Model version:** The exact YOLO version (for example, YOLOv8, YOLO11, or another Ultralytics-supported model) depends on the architecture used to train `best.pt`. Confirm the training configuration before specifying a particular version.

### Understanding the Output

The API returns unique class names rather than every individual detected object.

For example, if an image contains three banknotes classified as `100_taka`, the `notes` field contains that class name only once.

This means the current implementation identifies which banknote classes are present, but it does not return the number of instances of each denomination.

## 📊 Potential Applications

- Bangladeshi currency recognition systems.
- Currency identification prototypes for assistive technology.
- Computer vision and deep learning demonstrations.
- Image-based currency classification services.
- Integration with web applications or mobile application backends.

The project is a technical prototype and should not be treated as a verified financial authentication system.

## 🔒 Security and Limitations

The current implementation is suitable as a starting point for development and experimentation. Before production deployment, consider the following improvements:

- **Validate uploads:** Restrict file size and accepted image formats.
- **Handle invalid images:** Return a clear HTTP error for unreadable or malformed files.
- **Use safe file handling:** Consider processing images in memory and avoid trusting the original filename.
- **Improve response consistency:** Sort class names or return instance-level detections when required.
- **Configure the model path:** Use a path relative to the application file or a deployment environment variable.
- **Protect the endpoint:** Add authentication, rate limiting, and request-size limits if the API is publicly accessible.
- **Manage model resources:** Consider inference concurrency and available CPU/GPU memory.
- **Verify predictions:** Evaluate the model on diverse images, lighting conditions, note orientations, and backgrounds.

The current API does not independently verify banknote authenticity. It detects visual classes learned during training.

## 🛣️ Future Improvements

Possible future enhancements include:

- Returning confidence scores for each detection.
- Reporting the number of detected notes per denomination.
- Returning bounding-box coordinates.
- Supporting batch image inference.
- Building a frontend for uploading images and viewing results.
- Deploying the API to a cloud platform.
- Adding unit tests and API integration tests.
- Evaluating precision, recall, mAP@0.5, and mAP@0.5:0.95 on a held-out test dataset.
- Adding model versioning and reproducible dependency management.

## 🤝 Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Make and test your changes.
4. Commit your changes with a meaningful commit message.
5. Open a pull request describing the improvement.

## 👨‍💻 Author

**Fairuz Zahin Sneha**

- GitHub Profile: https://github.com/fairuzzahin

**Project Repository:** https://github.com/fairuzzahin/Bangladeshi_taka_Detection


---

*Developed using Python, FastAPI, Ultralytics YOLO, and Pillow for Bangladeshi Taka banknote detection.*
