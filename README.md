# Panorama Stitching and Homography

This project is a learning pack about **panorama stitching** and **homography** in computer vision. It explains how multiple images can be aligned and blended into a panorama using feature detection, feature matching, homography estimation, RANSAC, image warping, and blending.

The repository includes study material, presentation slides, a walkthrough notebook, further reading papers, a quiz, and an interactive Streamlit simulator.

## Project Overview

Panorama stitching is the process of combining overlapping images into one wider image. The core idea is to find matching points between images, estimate a geometric transformation called a **homography**, warp one image into the coordinate system of another image, and then blend the overlap so the final panorama looks smooth.

This project focuses on the main concepts behind that pipeline:

- Feature detection and description using methods such as SIFT and ORB
- Feature matching between overlapping images
- Homography estimation using matching points
- RANSAC for rejecting incorrect matches
- Image warping using a 3×3 transformation matrix
- Blending methods for reducing visible seams
- Failure cases such as parallax, rolling shutter distortion, moving objects, repeated patterns, and low-texture scenes

## Repository Structure

```text
Panorama_Stitching_and_Homography-main/
│
├── README.md
├── requirements.txt
├── walkthrough.ipynb
├── slides.pdf
├── slides.pptx
├── study_notes.pdf
├── study_notes.docx
├── quiz_with_rationale.pdf
├── quiz_with_rationale.docx
├── quiz_link.txt
│
├── simulator/
│   ├── app.py
│   ├── README.md
│   └── requirements.txt
│
└── further_reading/
    ├── A Comparative Analysis of Feature Detectors and Descriptors for Image Stitching.pdf
    ├── Automatic Panoramic Image Stitching using Invariant Features.pdf
    ├── Creating Panoramic Images Using ORB Feature Detection and RANSAC-based Image Alignment.pdf
    ├── Homography Estimation.pdf
    ├── ImageStitchingUsingHomographyTechniques.pdf
    └── Rolling Shutter Homography and its Applications.pdf
```

## Main Components

### 1. Study Notes

The study notes explain the theory behind panorama stitching and homography. They are useful for understanding the full pipeline before using the simulator or presenting the topic.

Files:

- `study_notes.pdf`
- `study_notes.docx`

### 2. Presentation Slides

The slides summarize the topic in a presentation-friendly format. They can be used to explain the technique, its pipeline, applications, and limitations.

Files:

- `slides.pdf`
- `slides.pptx`

### 3. Walkthrough Notebook

The notebook demonstrates important parts of the panorama stitching process using Python code.

File:

- `walkthrough.ipynb`

### 4. Interactive Simulator

The simulator is a Streamlit app called **Panorama Lab: Stitch, Warp, Fail, Fix**. It is designed to make the theory easier to understand through interactive visual examples.

The simulator includes:

- Pipeline overview
- Homography playground
- Pure rotation vs. parallax demonstration
- SIFT and feature matching demo
- RANSAC outlier lab
- Blending studio
- Failure mode gallery
- Interactive quiz

File:

- `simulator/app.py`

### 5. Quiz

The quiz checks understanding of the main concepts and includes rationales for the answers.

Files:

- `quiz_with_rationale.pdf`
- `quiz_with_rationale.docx`
- `quiz_link.txt`

### 6. Further Reading

The `further_reading/` folder contains papers and references used to support the project.

## Installation

Make sure Python is installed. Python 3.10 or newer is recommended.

Clone or download the project, then open a terminal inside the project folder.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Simulator

From the root project folder, run:

```bash
streamlit run simulator/app.py
```

Or go into the simulator folder and run:

```bash
cd simulator
streamlit run app.py
```

After running the command, Streamlit will open the app in your browser. If it does not open automatically, copy the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Running the Notebook

To open the walkthrough notebook, run:

```bash
jupyter notebook walkthrough.ipynb
```

Then open the notebook in your browser and run the cells step by step.

## Notes About OpenCV

The project uses `opencv-contrib-python-headless` because the simulator may use SIFT, which is included in the contrib version of OpenCV. If SIFT is not available on a system, the simulator is designed to fall back to ORB.

The headless OpenCV package is suitable for Streamlit apps because it does not require desktop GUI support.


## Project Type

This project is an educational computer vision project. It is intended to help students understand panorama stitching conceptually and practically through notes, slides, code, and interactive demonstrations.
