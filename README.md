#  Online Attendance System using KNN (Face Recognition)

A smart attendance management system that uses real-time face recognition powered by OpenCV and machine learning (K-Nearest Neighbors). No manual entry, no paperwork — just show your face and mark your attendance automatically.

---

##  Features

-  Face recognition using K-Nearest Neighbors (KNN)
-  Real-time webcam-based face detection
-  Automatic attendance saving to CSV with timestamp
-  Face data storage using Pickle and NumPy
-  Optional support for a background layout (`bg.png`)
-  Clean GUI overlay using OpenCV

---

## 🖥 Tech Stack

| Tech            | Usage                          |
|-----------------|--------------------------------|
| Python          | Main programming language      |
| OpenCV          | Image processing & webcam feed |
| NumPy           | Matrix operations              |
| Pickle          | Dataset serialization          |
| scikit-learn    | KNN classifier                 |
| CSV & OS Libs   | Attendance recording           |

---

##  Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/ShubhamDeshmuk-h/Online-Attendance-Using-KNN.git
cd Online-Attendance-Using-KNN

```

### 2. Install required packages
```bash
pip install -r requirements.txt
```
You can also manually install:

```bash

pip install opencv-python numpy scikit-learn
```

### 3. Prepare directories
Make sure the following folders exist:

```bash
/data              # For face_data.pkl and names.pkl
/Attendance        # Attendance CSVs will be saved here

```

### 4. Optional: Add background image

Place a file named bg.png in the root directory for GUI overlay (optional).


How It Works
Training Phase

Run your face data collection script to create face samples.

```bash
Dataset.py
```

Save them using pickle in data/ folder.

Recognition Phase

Launch the KNN-based recognition script.

It will recognize faces in real time and store attendance with a timestamp.

Running the Project
```bash

python attendance_knn.py
```
Press O to mark attendance for the recognized face.
Press Q to quit the application.


## Project Structure


├── attendance_knn.py<br/>
├── bg.png (optional)<br/>
├── /data/<br/>
│   ├── face_data.pkl<br/>
│   └── names.pkl<br/>
├── /Attendance/<br/>
│   └── Attendance_dd-mm-yyyy.csv<br/>
└── README.md
