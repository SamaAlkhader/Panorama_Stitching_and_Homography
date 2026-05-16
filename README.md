# Panorama Lab: Stitch, Warp, Fail, Fix

An interactive Streamlit simulator for the Computer Vision topic
**Panorama Stitching and Homography**.

It is designed as a teaching tool: every concept is paired with a slider, a
visual, and a short explanation so that students can build intuition by
experimenting.

## Features

The simulator contains eight tabs:

1. **Pipeline Overview** — the full panorama pipeline as a colored diagram,
   with what each step does, why it matters, and what can go wrong.
2. **Homography Playground** — adjust rotation, scale, translation, shear and
   perspective tilt and watch a 3×3 homography warp a checkerboard.
3. **Pure Rotation vs Parallax** — see why pure camera rotation works for one
   homography and why translation breaks it (depth-dependent shifts).
4. **SIFT and Feature Matching** — detect keypoints (SIFT if available, ORB
   fallback) on a synthetic scene and visualize good matches.
5. **RANSAC Outlier Lab** — generate matches with controllable outlier ratio
   and compare a least-squares fit to a RANSAC fit.
6. **Blending Studio** — try Hard Cut, Linear, Feathering and a multi-band
   style blend on two overlapping synthetic images.
7. **Failure Mode Gallery** — colorful cards covering parallax, rolling
   shutter, low texture, repeated patterns, moving objects, exposure
   differences and lens distortion, plus when not to use a single homography.
8. **Interactive Quiz** — 10 multiple-choice questions with explanations,
   score tracking and reset.

## Run locally

```bash
cd simulator
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Notes

- SIFT is preferred and is available through `opencv-contrib-python`. If for
  any reason it cannot be created on your system, the simulator falls back to
  ORB automatically.
- All images shown are generated procedurally in code — no datasets needed.
- Code uses English variable names and English comments throughout.
