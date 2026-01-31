# Cricket Score Detection Using Computer Vision

This project focuses on detecting and extracting **cricket score information** from images or video frames using **Computer Vision and Machine Learning techniques**.

The objective is to automatically identify score-related details (such as runs, overs, or scoreboard text) from visual inputs instead of manual observation.

The implementation is done using **Python**, without relying on heavy frameworks or end-to-end deep learning pipelines, keeping the project simple and educational.

---

## Project Objective

- Read cricket match images or frames
- Detect the scoreboard region
- Extract score information from the detected area
- Convert visual score data into readable text or numbers

This type of system can be used in:
- Sports analytics
- Broadcast automation
- Match data analysis
- Learning OCR and computer vision concepts

---

## Key Concepts Used

- Computer Vision
- Image Processing
- Optical Character Recognition (OCR)
- Feature extraction
- Basic Machine Learning logic

---

## Tech Stack

Programming Language:
- Python

Libraries Used:
- OpenCV
- NumPy
- Matplotlib
- pytesseract (OCR)
- Other supporting Python libraries

---

## How the System Works

1. Load cricket match image or video frame
2. Preprocess image (grayscale, thresholding, noise removal)
3. Detect scoreboard region
4. Apply OCR to extract text
5. Process extracted text to identify score values
6. Display or print detected score

---

## Project Structure

Cricket_score_detection/

|
|-- cricket_score_detection.py

|-- apps.py/

|-- cricket_preprocessed.xlsb

|-- README.md

---

## How to Run the Project

Step 1: Clone the repository

git clone https://github.com/mukesh-kumar-git/Cricket_score_detection.git  
cd Cricket_score_detection

Step 2: Install required libraries

pip install opencv-python numpy matplotlib pytesseract

Note:
Tesseract OCR must be installed separately on your system.

Step 3: Run the Python script

python cricket_score_detection.py

---

## Output

- Detects scoreboard area from cricket images
- Extracts score text using OCR
- Displays detected score information
- Helps understand how visual data can be converted into structured data

---

## Limitations

- Accuracy depends on image quality
- Works best with clear and readable scoreboards
- OCR performance may vary with font and lighting
- Not designed for real-time live broadcast usage

---

## Future Improvements

- Improve OCR accuracy with better preprocessing
- Support real-time video streams
- Use deep learning-based text detection
- Detect more match details (team names, wickets, overs)
- Create a GUI or web interface

---

## Purpose of This Project

- Learn computer vision fundamentals
- Practice OCR with real-world data
- Understand image preprocessing techniques
- Build a sports analytics–related ML project

---

## Author

Mukesh Kumar TM  
Electronics and Communication Engineering  
Python | Machine Learning 

---

## Note

This project is developed for learning and experimentation.
The focus is on understanding the pipeline rather than achieving production-level accuracy.
