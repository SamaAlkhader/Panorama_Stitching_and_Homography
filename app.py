"""
Panorama Lab: Stitch, Warp, Fail, Fix
=====================================

An interactive Streamlit simulator for the Computer Vision topic
"Panorama Stitching and Homography".

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


# ---------------------------------------------------------------------------
# Page config and theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Panorama Lab: Stitch, Warp, Fail, Fix",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


CUSTOM_CSS = """
<style>
/* Page background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
    color: #f1f5f9;
}

/* Gradient title */
.gradient-title {
    font-size: 3.0rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f472b6 0%, #facc15 30%, #34d399 60%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    line-height: 1.1;
}

.subtitle {
    font-size: 1.1rem;
    color: #cbd5e1;
    margin-bottom: 1.5rem;
}

/* Card style */
.card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(6px);
}

.card h3, .card h4 {
    margin-top: 0;
    color: #fde68a;
}

.card p, .card li {
    color: #e2e8f0;
    line-height: 1.5;
}

.section-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #fcd34d;
    margin-top: 0.4rem;
    margin-bottom: 0.4rem;
}

.caption {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-bottom: 1rem;
}

/* Pipeline pills */
.pipeline-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    align-items: center;
    margin: 0.6rem 0 1.4rem 0;
}

.pipeline-pill {
    padding: 0.6rem 1.0rem;
    border-radius: 999px;
    color: #0f172a;
    font-weight: 700;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}

.pipeline-arrow {
    color: #facc15;
    font-size: 1.4rem;
    font-weight: 800;
}

/* Warning card */
.warn-card {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.25), rgba(249, 115, 22, 0.25));
    border: 1px solid rgba(248, 113, 113, 0.6);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    color: #fff7ed;
    font-weight: 600;
}

/* Success card */
.ok-card {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.22), rgba(16, 185, 129, 0.22));
    border: 1px solid rgba(74, 222, 128, 0.55);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    color: #ecfdf5;
    font-weight: 600;
}

/* Failure card variants */
.fail-card {
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
    color: #0f172a;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3);
}
.fail-card h4 {
    margin-top: 0;
    margin-bottom: 0.5rem;
    color: #0f172a;
}
.fail-card p { margin: 0.25rem 0; }

/* Quiz */
.quiz-card {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1.1rem;
}
.quiz-correct {
    background: rgba(34, 197, 94, 0.18);
    border: 1px solid #4ade80;
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    color: #ecfdf5;
}
.quiz-wrong {
    background: rgba(239, 68, 68, 0.18);
    border: 1px solid #f87171;
    border-radius: 10px;
    padding: 0.6rem 0.9rem;
    color: #fff1f2;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px 10px 0 0;
    color: #e2e8f0;
    font-weight: 600;
    padding: 0.6rem 1rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #ec4899);
    color: #ffffff !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="gradient-title">Panorama Lab: Stitch, Warp, Fail, Fix</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">An interactive playground for understanding panorama '
    "stitching and homography in Computer Vision.</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def make_checkerboard(size: int = 360, squares: int = 8) -> np.ndarray:
    """Generate a colorful checkerboard image with a small marker for orientation."""
    step = size // squares
    img = np.zeros((size, size, 3), dtype=np.uint8)
    palette = [
        (244, 114, 182),  # pink
        (250, 204, 21),   # yellow
        (52, 211, 153),   # green
        (56, 189, 248),   # sky
        (167, 139, 250),  # violet
        (248, 113, 113),  # red
    ]
    for i in range(squares):
        for j in range(squares):
            color = palette[(i + j) % len(palette)]
            img[i * step:(i + 1) * step, j * step:(j + 1) * step] = color

    # Orientation marker: small white triangle in the top-left square
    cv2.circle(img, (step // 2, step // 2), step // 4, (255, 255, 255), -1)
    cv2.putText(
        img, "A", (step + 10, step + step // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (15, 23, 42), 2, cv2.LINE_AA,
    )
    return img


def make_synthetic_scene(width: int = 480, height: int = 320, seed: int = 7) -> np.ndarray:
    """Generate a synthetic scene with corners, circles, text and shapes for SIFT."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 245, dtype=np.uint8)

    # Background gradient
    for y in range(height):
        shade = int(220 + 25 * (y / height))
        img[y, :] = (shade, shade - 10, shade - 20)

    # Random colorful rectangles
    for _ in range(8):
        x1, y1 = rng.integers(0, width - 60), rng.integers(0, height - 60)
        w, h = rng.integers(30, 80), rng.integers(30, 80)
        color = tuple(int(c) for c in rng.integers(40, 230, size=3))
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, -1)

    # Circles
    for _ in range(6):
        cx, cy = rng.integers(20, width - 20), rng.integers(20, height - 20)
        r = rng.integers(10, 35)
        color = tuple(int(c) for c in rng.integers(40, 230, size=3))
        cv2.circle(img, (int(cx), int(cy)), int(r), color, -1)
        cv2.circle(img, (int(cx), int(cy)), int(r), (15, 23, 42), 2)

    # Strong corners
    cv2.rectangle(img, (40, 40), (110, 110), (15, 23, 42), 3)
    cv2.rectangle(img, (width - 130, height - 110), (width - 40, height - 40), (15, 23, 42), 3)

    # Text features
    cv2.putText(img, "PANO", (60, 200), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, (30, 30, 80), 3, cv2.LINE_AA)
    cv2.putText(img, "LAB", (width - 200, 80), cv2.FONT_HERSHEY_DUPLEX,
                1.2, (120, 20, 80), 2, cv2.LINE_AA)

    # Diagonal stripes for texture
    for k in range(-height, width, 30):
        cv2.line(img, (k, 0), (k + height, height), (200, 200, 220), 1)

    return img


def make_two_overlapping_images(
    width: int = 480,
    height: int = 320,
    overlap: int = 160,
    brightness_diff: int = 0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return two images of the same total size that overlap in the central region.

    Image A has content on the left half, Image B on the right half.
    The overlap region is `overlap` pixels wide and centered.
    """
    img_a = np.full((height, width, 3), 30, dtype=np.uint8)
    img_b = np.full((height, width, 3), 30, dtype=np.uint8)

    # Image A: a warm gradient with shapes on the left
    for x in range(width):
        t = x / max(1, width - 1)
        col = (int(30 + 200 * (1 - t)), int(60 + 100 * (1 - t)), int(180 - 100 * t))
        img_a[:, x] = col
    cv2.circle(img_a, (120, 100), 50, (255, 230, 120), -1)
    cv2.circle(img_a, (120, 100), 50, (40, 40, 40), 2)
    cv2.rectangle(img_a, (60, 200), (200, 280), (250, 200, 80), -1)
    cv2.putText(img_a, "A", (90, 255), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (40, 20, 60), 3, cv2.LINE_AA)

    # Image B: a cool gradient with shapes on the right
    for x in range(width):
        t = x / max(1, width - 1)
        col = (int(220 - 120 * (1 - t)), int(120 + 100 * t), int(80 + 160 * t))
        img_b[:, x] = col
    cv2.circle(img_b, (width - 120, height - 100), 55, (120, 220, 255), -1)
    cv2.circle(img_b, (width - 120, height - 100), 55, (40, 40, 40), 2)
    cv2.rectangle(img_b, (width - 220, 50), (width - 60, 140), (90, 220, 200), -1)
    cv2.putText(img_b, "B", (width - 160, 110), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (20, 40, 60), 3, cv2.LINE_AA)

    # Apply brightness offset to image B
    if brightness_diff != 0:
        img_b = np.clip(img_b.astype(np.int16) + brightness_diff, 0, 255).astype(np.uint8)

    return img_a, img_b, overlap


def build_homography(
    rotation_deg: float,
    scale: float,
    tx: float,
    ty: float,
    shear: float,
    p_x: float,
    p_y: float,
    center: Tuple[float, float],
) -> np.ndarray:
    """Build a 3x3 homography matrix from intuitive parameters around `center`."""
    cx, cy = center
    # Translate to origin
    t1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    # Rotation + scale + shear (affine part)
    theta = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    affine = np.array([
        [scale * cos_t, scale * (-sin_t + shear), 0.0],
        [scale * sin_t, scale * cos_t, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    # Perspective tilt
    persp = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [p_x, p_y, 1.0],
    ], dtype=np.float64)
    # Translate back and apply user translation
    t2 = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], dtype=np.float64)

    h = t2 @ persp @ affine @ t1
    return h


def warp_with_homography(img: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Warp `img` with homography `h` keeping the same output size."""
    height, width = img.shape[:2]
    return cv2.warpPerspective(
        img, h, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(15, 23, 42),
    )


def get_feature_detector(max_features: int):
    """Try to create a SIFT detector; fall back to ORB if SIFT is unavailable."""
    try:
        sift = cv2.SIFT_create(nfeatures=max_features)
        return sift, "SIFT"
    except Exception:
        orb = cv2.ORB_create(nfeatures=max_features)
        return orb, "ORB"


def detect_and_match(
    img_a: np.ndarray,
    img_b: np.ndarray,
    max_features: int,
    ratio: float,
):
    """Detect features in both images and return keypoints + good matches."""
    detector, name = get_feature_detector(max_features)
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    kp_a, des_a = detector.detectAndCompute(gray_a, None)
    kp_b, des_b = detector.detectAndCompute(gray_b, None)

    good: List[cv2.DMatch] = []
    if des_a is not None and des_b is not None and len(kp_a) > 1 and len(kp_b) > 1:
        if name == "SIFT":
            matcher = cv2.BFMatcher(cv2.NORM_L2)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn = matcher.knnMatch(des_a, des_b, k=2)
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return kp_a, kp_b, good, name


def draw_keypoints(img: np.ndarray, kp) -> np.ndarray:
    """Draw colored keypoints on top of image."""
    vis = img.copy()
    for k in kp:
        x, y = int(k.pt[0]), int(k.pt[1])
        cv2.circle(vis, (x, y), 4, (255, 215, 64), 1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (x, y), 1, (236, 72, 153), -1)
    return vis


def draw_matches_image(img_a, kp_a, img_b, kp_b, matches) -> np.ndarray:
    """Wrapper around cv2.drawMatches with nicer defaults. Sorts by distance first."""
    top = sorted(matches, key=lambda m: m.distance)[:60]
    return cv2.drawMatches(
        img_a, kp_a, img_b, kp_b, top, None,
        matchColor=(80, 220, 120),
        singlePointColor=(244, 114, 182),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def draw_matches_with_inliers(
    img_a: np.ndarray, kp_a, img_b: np.ndarray, kp_b,
    matches, inlier_mask, max_draw: int = 60,
) -> np.ndarray:
    """Draw matches with inliers in green and outliers in red, side by side."""
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]
    h_out = max(h_a, h_b)
    canvas = np.zeros((h_out, w_a + w_b, 3), dtype=np.uint8)
    canvas[:h_a, :w_a] = img_a
    canvas[:h_b, w_a:w_a + w_b] = img_b

    pairs = list(zip(matches, inlier_mask.flatten().tolist()
                     if inlier_mask is not None
                     else [1] * len(matches)))
    # Sort by distance and clamp the number drawn.
    pairs.sort(key=lambda p: p[0].distance)
    pairs = pairs[:max_draw]

    for m, is_in in pairs:
        pa = kp_a[m.queryIdx].pt
        pb = kp_b[m.trainIdx].pt
        x1, y1 = int(pa[0]), int(pa[1])
        x2, y2 = int(pb[0]) + w_a, int(pb[1])
        color = (80, 220, 120) if is_in else (240, 80, 80)
        cv2.line(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 3, color, -1)
        cv2.circle(canvas, (x2, y2), 3, color, -1)
    return canvas


def generate_ransac_data(
    n_matches: int, outlier_pct: float, noise: float, seed: int = 0
):
    """Generate synthetic point matches that follow a known affine map plus outliers."""
    rng = np.random.default_rng(seed)
    n_outliers = int(n_matches * outlier_pct / 100.0)
    n_inliers = n_matches - n_outliers

    # Ground-truth affine-ish transform (a homography with no perspective)
    theta = math.radians(15.0)
    s = 1.1
    h_true = np.array([
        [s * math.cos(theta), -s * math.sin(theta), 30.0],
        [s * math.sin(theta),  s * math.cos(theta), -10.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    pts_src = rng.uniform(20, 280, size=(n_inliers, 2))
    pts_homo = np.hstack([pts_src, np.ones((n_inliers, 1))])
    pts_dst = (h_true @ pts_homo.T).T[:, :2]
    pts_dst += rng.normal(0, noise, size=pts_dst.shape)

    # Outliers: random destination points
    out_src = rng.uniform(20, 280, size=(n_outliers, 2))
    out_dst = rng.uniform(20, 280, size=(n_outliers, 2))

    src = np.vstack([pts_src, out_src]).astype(np.float32)
    dst = np.vstack([pts_dst, out_dst]).astype(np.float32)
    truth_mask = np.array([1] * n_inliers + [0] * n_outliers)

    # Shuffle so order does not leak the truth
    perm = rng.permutation(len(src))
    return src[perm], dst[perm], truth_mask[perm], h_true


def linear_blend(a: np.ndarray, b: np.ndarray, mask_a: np.ndarray) -> np.ndarray:
    """Blend two RGB images using a single-channel alpha mask in [0, 1]."""
    m = mask_a[:, :, None]
    out = a.astype(np.float32) * m + b.astype(np.float32) * (1 - m)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_alpha_mask(
    height: int,
    width: int,
    seam: int,
    overlap: int,
    method: str,
    blur: int,
) -> np.ndarray:
    """Create an alpha mask for image A based on the chosen blending method."""
    mask = np.zeros((height, width), dtype=np.float32)
    half = max(1, overlap // 2)
    left = max(0, seam - half)
    right = min(width, seam + half)

    if method == "Hard Cut":
        mask[:, :seam] = 1.0
    elif method == "Linear Blend":
        mask[:, :left] = 1.0
        if right > left:
            ramp = np.linspace(1.0, 0.0, right - left, dtype=np.float32)
            mask[:, left:right] = ramp[None, :]
    elif method == "Feathering":
        mask[:, :left] = 1.0
        if right > left:
            x = np.linspace(0, 1, right - left, dtype=np.float32)
            ramp = 0.5 * (1 + np.cos(math.pi * x))  # smooth cosine
            mask[:, left:right] = ramp[None, :]
    else:  # Multi-band-like
        mask[:, :seam] = 1.0
        k = max(1, blur) | 1  # ensure odd
        mask = cv2.GaussianBlur(mask, (k * 2 + 1, k * 2 + 1), 0)
    return mask


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def init_quiz_state() -> None:
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    if "quiz_checked" not in st.session_state:
        st.session_state.quiz_checked = {}


def reset_quiz() -> None:
    st.session_state.quiz_answers = {}
    st.session_state.quiz_checked = {}


init_quiz_state()

tab_overview, tab_homography, tab_parallax, tab_sift, tab_ransac, \
    tab_blending, tab_failures, tab_full, tab_quiz = st.tabs([
        "Pipeline Overview",
        "Homography Playground",
        "Pure Rotation vs Parallax",
        "SIFT and Feature Matching",
        "RANSAC Outlier Lab",
        "Blending Studio",
        "Failure Mode Gallery",
        "Full Stitching Demo",
        "Interactive Quiz",
    ])


# ---------------------------------------------------------------------------
# Tab 1: Pipeline Overview
# ---------------------------------------------------------------------------

with tab_overview:
    st.markdown('<div class="section-title">The Panorama Stitching Pipeline</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="caption">A panorama is built in stages. Each stage solves '
        "one well-defined problem and hands its output to the next.</div>",
        unsafe_allow_html=True,
    )

    pipeline_steps = [
        ("Input Images", "#f472b6"),
        ("SIFT Features", "#facc15"),
        ("Feature Matching", "#fb923c"),
        ("RANSAC", "#34d399"),
        ("Homography Warp", "#38bdf8"),
        ("Blending", "#a78bfa"),
        ("Final Panorama", "#f87171"),
    ]
    pills_html = '<div class="pipeline-row">'
    for i, (name, color) in enumerate(pipeline_steps):
        pills_html += (
            f'<span class="pipeline-pill" style="background:{color};">{name}</span>'
        )
        if i < len(pipeline_steps) - 1:
            pills_html += '<span class="pipeline-arrow">&#10148;</span>'
    pills_html += "</div>"
    st.markdown(pills_html, unsafe_allow_html=True)

    step_details = [
        ("Input Images",
         "Two or more overlapping photographs of the same scene.",
         "All later steps assume the inputs share enough scene content to be matched.",
         "Too little overlap, very different exposures, or moving objects."),
        ("SIFT Features",
         "Detect distinctive keypoints and compute descriptors that are robust to scale and rotation.",
         "Stable, repeatable features are the anchors that make matching possible.",
         "Low-texture areas (sky, plain walls) produce few keypoints."),
        ("Feature Matching",
         "Compare descriptors between the two images and keep the best correspondences.",
         "Each match is a hypothesis that says: this point in image A is the same physical point in image B.",
         "Repeated patterns (windows, tiles) cause many ambiguous matches."),
        ("RANSAC",
         "Repeatedly fit a homography from random subsets and count agreeing matches (inliers).",
         "Filters out wrong matches that would otherwise corrupt the geometry.",
         "If the true inlier ratio is too low, RANSAC may pick a bad model."),
        ("Homography Warp",
         "Apply the estimated 3x3 matrix with cv2.warpPerspective to align one image to the other.",
         "Brings the two images into a common coordinate frame.",
         "A homography only models pure rotation or planar scenes; translation breaks the assumption."),
        ("Blending",
         "Combine pixels in the overlap region (linear, feathering, multi-band).",
         "Hides the seam and exposure differences so the result looks like a single photograph.",
         "Misalignment leaves visible ghosting that no blending can completely hide."),
        ("Final Panorama",
         "The merged wide-angle image.",
         "This is the output the user actually sees.",
         "Quality of every previous step shows up here."),
    ]

    cols = st.columns(2)
    for i, (name, what, why, fail) in enumerate(step_details):
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="card">
                    <h3>{i + 1}. {name}</h3>
                    <p><b>What it does:</b> {what}</p>
                    <p><b>Why it matters:</b> {why}</p>
                    <p><b>What can go wrong:</b> {fail}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Tab 2: Homography Playground
# ---------------------------------------------------------------------------

with tab_homography:
    st.markdown('<div class="section-title">Homography Playground</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="caption">Move the sliders to build a 3x3 homography matrix '
        "and watch how it warps the colorful checkerboard.</div>",
        unsafe_allow_html=True,
    )

    ctrl_col, view_col = st.columns([1, 2])
    with ctrl_col:
        st.markdown('<div class="card"><h4>Transform Controls</h4>', unsafe_allow_html=True)
        rotation = st.slider("Rotation angle (deg)", -90.0, 90.0, 15.0, 1.0)
        scale = st.slider("Scale", 0.3, 2.0, 1.0, 0.05)
        tx = st.slider("Translation X", -150, 150, 0, 1)
        ty = st.slider("Translation Y", -150, 150, 0, 1)
        shear = st.slider("Shear", -0.6, 0.6, 0.0, 0.01)
        p_x = st.slider("Perspective tilt X", -0.0015, 0.0015, 0.0, 0.0001, format="%.4f")
        p_y = st.slider("Perspective tilt Y", -0.0015, 0.0015, 0.0, 0.0001, format="%.4f")
        st.markdown("</div>", unsafe_allow_html=True)

    img = make_checkerboard(360, 8)
    h = build_homography(rotation, scale, tx, ty, shear, p_x, p_y,
                         center=(img.shape[1] / 2, img.shape[0] / 2))
    warped = warp_with_homography(img, h)

    with view_col:
        c1, c2 = st.columns(2)
        c1.image(img, caption="Original image", use_container_width=True)
        c2.image(warped, caption="Warped image (cv2.warpPerspective)",
                 use_container_width=True)

        st.markdown('<div class="card"><h4>Current Homography Matrix</h4>',
                    unsafe_allow_html=True)
        h_norm = h / h[2, 2]
        st.dataframe(
            pd.DataFrame(
                {
                    "col 0": [f"{h_norm[i, 0]: .4f}" for i in range(3)],
                    "col 1": [f"{h_norm[i, 1]: .4f}" for i in range(3)],
                    "col 2": [f"{h_norm[i, 2]: .4f}" for i in range(3)],
                },
                index=["h0", "h1", "h2"],
            ),
            use_container_width=True,
        )
        st.markdown(
            "<p>A homography is a 3x3 matrix, so it has 9 entries — but it lives in "
            "homogeneous coordinates, so the overall scale does not matter. We can "
            "always divide by one non-zero entry, leaving exactly <b>8 degrees of "
            "freedom</b>.</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="card">
                <h4>How to read the matrix</h4>
                <ul>
                    <li><b>Top-left 2x2</b> (h00, h01, h10, h11): rotation + scale + shear.</li>
                    <li><b>Top-right column</b> (h02, h12): translation (tx, ty).</li>
                    <li><b>Bottom row</b> (h20, h21): perspective distortion. h22 is the overall scale and is normalised to 1.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 3: Pure Rotation vs Parallax
# ---------------------------------------------------------------------------

with tab_parallax:
    st.markdown('<div class="section-title">Pure Rotation vs Parallax</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="caption">A single homography assumes the camera rotates about '
        "its optical center. Translation introduces depth-dependent shifts that "
        "no single homography can correct.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    rot_angle = c1.slider("Camera rotation (deg)", -30.0, 30.0, 8.0, 0.5)
    cam_trans = c2.slider("Camera translation", 0.0, 60.0, 10.0, 1.0)
    depth_diff = c3.slider("Depth difference (near vs far)", 0.0, 1.0, 0.6, 0.05)

    def render_scene(translation: float) -> np.ndarray:
        """Render two views of near and far objects under camera rot+trans."""
        width, height = 520, 320
        canvas = np.full((height, width, 3), 30, dtype=np.uint8)

        # "Far" objects shift little, "near" objects shift a lot under translation.
        far_shift = translation * (1 - depth_diff) * 0.4
        near_shift = translation * (1 + depth_diff) * 1.6

        # Apply rotation as a small skew of horizon
        horizon = int(height * 0.55 + rot_angle * 1.2)
        cv2.rectangle(canvas, (0, 0), (width, horizon), (90, 130, 200), -1)   # sky
        cv2.rectangle(canvas, (0, horizon), (width, height), (60, 110, 70), -1)  # ground

        # Far mountains
        for i, x in enumerate([60, 180, 300, 420]):
            x2 = int(x + far_shift)
            pts = np.array([[x2 - 50, horizon], [x2, horizon - 60 - i * 5],
                            [x2 + 60, horizon]], dtype=np.int32)
            cv2.fillPoly(canvas, [pts], (130, 130, 160))

        # Near trees
        for i, x in enumerate([90, 230, 370]):
            x2 = int(x + near_shift)
            cv2.rectangle(canvas, (x2 - 8, horizon), (x2 + 8, horizon + 70),
                          (90, 60, 30), -1)
            cv2.circle(canvas, (x2, horizon - 10), 30, (40, 160, 70), -1)

        # Apply rotation as image rotation
        m_rot = cv2.getRotationMatrix2D((width / 2, height / 2), rot_angle, 1.0)
        return cv2.warpAffine(canvas, m_rot, (width, height),
                              borderValue=(30, 30, 30))

    pure_rotation_view = render_scene(translation=0.0)
    translation_view = render_scene(translation=cam_trans)

    col_a, col_b = st.columns(2)
    col_a.image(pure_rotation_view,
                caption="Pure rotation: one homography aligns everything",
                use_container_width=True)
    col_b.image(translation_view,
                caption="With translation: near objects shift more than far objects",
                use_container_width=True)

    if cam_trans > 25:
        st.markdown(
            '<div class="warn-card">High parallax risk: one homography cannot align '
            "all depths correctly.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="ok-card">Translation is small. A homography is a reasonable '
            "approximation.</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="card">
            <h4>Why this happens</h4>
            <ul>
                <li>Pure rotation about the optical center maps all 3D rays the same way regardless of depth, so a single homography is exact.</li>
                <li>Translation moves the camera through space. The amount each pixel shifts depends on the depth of the object it represents.</li>
                <li>Near objects shift more than far objects. This is parallax.</li>
                <li>Trying to align both with a single homography produces ghosting on whichever depth was not chosen as the reference.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Real failure demo: force a homography on a parallax scene ----
    st.markdown(
        '<div class="section-title">What happens when you force a homography '
        "on a parallax scene?</div>",
        unsafe_allow_html=True,
    )

    def render_parallax_scene(shift_far: float, shift_near: float) -> np.ndarray:
        """Render a 2-depth scene: a flat background rectangle (far) plus a
        circle (near). Shifts are applied independently to simulate parallax."""
        width, height = 480, 320
        scene = np.full((height, width, 3), 30, dtype=np.uint8)
        # Far layer: a coloured rectangle that fills most of the frame.
        fx = int(shift_far)
        cv2.rectangle(scene, (40 + fx, 50), (440 + fx, 270),
                      (90, 140, 220), -1)
        cv2.rectangle(scene, (40 + fx, 50), (440 + fx, 270),
                      (40, 60, 110), 3)
        cv2.putText(scene, "FAR", (60 + fx, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (20, 30, 70), 2, cv2.LINE_AA)
        # Near layer: a bright circle drawn on top.
        nx = int(shift_near)
        cv2.circle(scene, (240 + nx, 180), 55, (250, 220, 120), -1)
        cv2.circle(scene, (240 + nx, 180), 55, (30, 30, 30), 2)
        cv2.putText(scene, "NEAR", (210 + nx, 188), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (40, 30, 20), 2, cv2.LINE_AA)
        return scene

    # Drive shifts from the existing sliders.
    near_shift = cam_trans * (1 + depth_diff) * 1.6
    far_shift = cam_trans * (1 - depth_diff) * 0.4

    view_a = render_parallax_scene(shift_far=0.0, shift_near=0.0)
    view_b = render_parallax_scene(shift_far=far_shift, shift_near=near_shift)

    # Synthetic corner correspondences for the FAR rectangle only.
    far_corners_a = np.array([
        [40.0, 50.0], [440.0, 50.0], [440.0, 270.0], [40.0, 270.0],
    ], dtype=np.float32)
    far_corners_b = far_corners_a + np.array([[far_shift, 0.0]], dtype=np.float32)
    h_far, _ = cv2.findHomography(far_corners_b, far_corners_a, method=0)

    if h_far is None:
        h_far = np.eye(3, dtype=np.float64)
    warped_b = cv2.warpPerspective(
        view_b, h_far, (view_a.shape[1], view_a.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=(30, 30, 30),
    )
    forced = cv2.addWeighted(view_a, 0.5, warped_b, 0.5, 0.0)

    p1, p2, p3 = st.columns(3)
    p1.image(view_a, caption="View A (reference)", use_container_width=True)
    p2.image(view_b, caption="View B (camera translated)",
             use_container_width=True)
    p3.image(forced, caption="Forced alignment (ghosting)",
             use_container_width=True)

    st.markdown(
        """
        <div class="card">
            <p>The far background aligns correctly because the homography was
            fitted to it. The near circle ghosts because its shift was larger —
            no single 3x3 matrix can correct both depths at once. This is the
            core failure of homography-based stitching when the camera
            translates.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab 4: SIFT and Feature Matching
# ---------------------------------------------------------------------------

with tab_sift:
    st.markdown('<div class="section-title">SIFT and Feature Matching</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="caption">Detect keypoints in a synthetic scene and its '
        "transformed copy. SIFT is preferred; ORB is used automatically as a "
        "fallback if SIFT is not available.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    max_features = c1.slider("Maximum number of features", 50, 1500, 500, 50)
    c1.caption("More features = more potential matches but slower")
    ratio = c2.slider("Match ratio threshold (Lowe)", 0.5, 0.95, 0.75, 0.01)
    c2.caption("Lower = stricter Lowe filter, fewer but better matches")
    strength = c3.slider("Transformation strength", 0.0, 1.0, 0.4, 0.05)
    c3.caption("How much the second image is warped; higher = harder to match")
    sift_ransac_thr = c4.slider("RANSAC reproj threshold", 1.0, 20.0, 5.0, 0.5)
    c4.caption("Max pixel error allowed for a match to be called an inlier")

    base = make_synthetic_scene(480, 320)
    h_demo = build_homography(
        rotation_deg=20 * strength,
        scale=1.0 + 0.2 * strength,
        tx=40 * strength,
        ty=-20 * strength,
        shear=0.05 * strength,
        p_x=0.0006 * strength,
        p_y=0.0003 * strength,
        center=(base.shape[1] / 2, base.shape[0] / 2),
    )
    transformed = warp_with_homography(base, h_demo)

    kp_a, kp_b, good, det_name = detect_and_match(base, transformed, max_features, ratio)

    vis_a = draw_keypoints(base, kp_a)
    vis_b = draw_keypoints(transformed, kp_b)

    # Run RANSAC on top of Lowe-filtered matches.
    inlier_mask_sift = None
    H_sift = None
    n_in_sift = 0
    det_H = float("nan")
    if len(good) >= 4:
        src_pts = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H_sift, inlier_mask_sift = cv2.findHomography(
            src_pts, dst_pts, cv2.RANSAC, sift_ransac_thr,
        )
        if inlier_mask_sift is not None:
            n_in_sift = int(inlier_mask_sift.sum())
        if H_sift is not None:
            det_H = float(np.linalg.det(H_sift))

    if good:
        matches_img = draw_matches_with_inliers(
            base, kp_a, transformed, kp_b, good, inlier_mask_sift, max_draw=60,
        )
    else:
        matches_img = np.hstack([base, transformed])

    col_l, col_r = st.columns(2)
    col_l.image(vis_a, caption=f"Original image with {len(kp_a)} keypoints",
                use_container_width=True)
    col_r.image(vis_b, caption=f"Transformed image with {len(kp_b)} keypoints",
                use_container_width=True)

    st.image(matches_img,
             caption=f"Matches (green = RANSAC inlier, red = outlier). "
                     f"Detector: {det_name}, total good: {len(good)}",
             use_container_width=True)

    n_out_sift = max(0, len(good) - n_in_sift)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total matches", len(good))
    m2.metric("RANSAC inliers", n_in_sift)
    m3.metric("RANSAC outliers", n_out_sift)
    m4.metric("det(H)", f"{det_H:.3f}" if not math.isnan(det_H) else "n/a")

    st.markdown(
        """
        <div class="card">
            <h4>Why SIFT works for stitching</h4>
            <ul>
                <li><b>Scale invariance:</b> the same physical point is recognized whether the camera is closer or farther.</li>
                <li><b>Rotation invariance:</b> features keep their identity even if the second photo is rotated relative to the first.</li>
                <li>Good correspondences are the foundation of homography estimation. Without them, RANSAC has no inliers to find.</li>
            </ul>
        </div>
        <div class="card">
            <p>Lowe's ratio test removes ambiguous matches. RANSAC then removes
            geometrically inconsistent ones. Only inliers (green) are used to
            estimate the homography. Outliers (red) are matches that survived
            Lowe's test but contradict the estimated camera motion — they would
            corrupt a least-squares fit. The determinant of H should be positive
            and close to 1 for a well-conditioned transform.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab 5: RANSAC Outlier Lab
# ---------------------------------------------------------------------------

with tab_ransac:
    st.markdown('<div class="section-title">RANSAC Outlier Lab</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="caption">Generate synthetic point matches with a known '
        "ground-truth transform, then add outliers and watch what happens to the "
        "estimated homography with and without RANSAC.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    n_matches = c1.slider("Number of matches", 20, 400, 120, 10)
    c1.caption("Total synthetic correspondences generated")
    outlier_pct = c2.slider("Outlier percentage", 0, 80, 35, 5)
    c2.caption("Fraction of matches with random wrong destinations")
    ransac_thr = c3.slider("RANSAC threshold (px)", 1.0, 20.0, 5.0, 0.5)
    c3.caption("A match is an inlier if its reprojection error < this value in pixels")
    n_iters = c4.slider("RANSAC iterations", 100, 5000, 2000, 100)
    c4.caption("More iterations = higher probability of finding a clean subset, but slower")

    src, dst, truth_mask, h_true = generate_ransac_data(
        n_matches, outlier_pct, noise=1.5
    )

    # Naive least-squares (no RANSAC) using all matches.
    h_naive, _ = cv2.findHomography(src, dst, method=0)
    # RANSAC
    h_ransac, mask_ransac = cv2.findHomography(
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thr, maxIters=n_iters, confidence=0.99,
    )

    inliers = int(mask_ransac.sum()) if mask_ransac is not None else 0

    def reprojection_error(h: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray,
                           inlier_only_mask: Optional[np.ndarray] = None) -> float:
        if h is None:
            return float("nan")
        ones = np.ones((len(src_pts), 1))
        homo = np.hstack([src_pts, ones])
        proj = (h @ homo.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        err = np.linalg.norm(proj - dst_pts, axis=1)
        if inlier_only_mask is not None:
            sel = inlier_only_mask.astype(bool)
            if sel.sum() == 0:
                return float("nan")
            return float(err[sel].mean())
        return float(err.mean())

    # Errors measured against the ground-truth inliers only (fair comparison).
    truth_bool = truth_mask.astype(bool)
    err_naive = reprojection_error(h_naive, src, dst, inlier_only_mask=truth_mask)
    err_ransac = reprojection_error(h_ransac, src, dst, inlier_only_mask=truth_mask)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.set_facecolor("#0f172a")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#475569")

    # Left: all matches as displacement vectors
    axes[0].set_title("All matches", color="#fde68a")
    for s, d, t in zip(src, dst, truth_mask):
        color = "#34d399" if t == 1 else "#f87171"
        axes[0].plot([s[0], d[0]], [s[1], d[1]], color=color, alpha=0.5, lw=0.8)
        axes[0].scatter(s[0], s[1], color=color, s=14, marker="o", edgecolor="white", lw=0.3)
        axes[0].scatter(d[0], d[1], color=color, s=14, marker="x")
    axes[0].invert_yaxis()
    axes[0].set_aspect("equal")
    axes[0].set_xlim(-20, 320)
    axes[0].set_ylim(320, -20)
    axes[0].legend(handles=[
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#34d399",
                   markersize=8, label="True inlier", linestyle=""),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#f87171",
                   markersize=8, label="Outlier", linestyle=""),
    ], facecolor="#1e293b", labelcolor="#e2e8f0", loc="upper right")

    # Right: RANSAC classification
    axes[1].set_title("RANSAC inliers vs outliers", color="#fde68a")
    if mask_ransac is not None:
        for s, d, m in zip(src, dst, mask_ransac.flatten()):
            color = "#38bdf8" if m == 1 else "#f59e0b"
            marker = "o" if m == 1 else "X"
            axes[1].scatter(s[0], s[1], color=color, s=22, marker=marker,
                            edgecolor="white", lw=0.3)
    axes[1].invert_yaxis()
    axes[1].set_aspect("equal")
    axes[1].set_xlim(-20, 320)
    axes[1].set_ylim(320, -20)
    axes[1].legend(handles=[
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#38bdf8",
                   markersize=9, label="RANSAC inlier", linestyle=""),
        plt.Line2D([0], [0], marker="X", color="w", markerfacecolor="#f59e0b",
                   markersize=9, label="RANSAC outlier", linestyle=""),
    ], facecolor="#1e293b", labelcolor="#e2e8f0", loc="upper right")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=110)
    plt.close(fig)
    st.image(buf.getvalue(), use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total matches", len(src))
    m2.metric("True inliers", int(truth_mask.sum()))
    m3.metric("RANSAC inliers", inliers)
    m4.metric("Outlier rate (set)", f"{outlier_pct}%")

    e1, e2 = st.columns(2)
    e1.metric("Mean reprojection error (least-squares)",
              f"{err_naive:.2f} px" if not math.isnan(err_naive) else "n/a")
    e2.metric("Mean reprojection error (RANSAC)",
              f"{err_ransac:.2f} px" if not math.isnan(err_ransac) else "n/a")

    st.markdown(
        """
        <div class="card">
            <p>Mean reprojection error measures how far the estimated homography
            projects source points away from their known destinations. A lower
            error means a more accurate alignment. RANSAC's error is computed
            only on inliers; the naive error is computed on all matches
            including outliers, which inflates it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Visual impact on image alignment ----
    st.markdown('<div class="section-title">Visual impact on image alignment</div>',
                unsafe_allow_html=True)
    src_img = make_checkerboard(200, 6)
    h_w, w_w = src_img.shape[:2]
    if h_naive is not None:
        warp_naive = cv2.warpPerspective(
            src_img, h_naive, (w_w, h_w),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(15, 23, 42),
        )
    else:
        warp_naive = np.zeros_like(src_img)
    if h_ransac is not None:
        warp_ransac = cv2.warpPerspective(
            src_img, h_ransac, (w_w, h_w),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(15, 23, 42),
        )
    else:
        warp_ransac = np.zeros_like(src_img)

    w1, w2, w3 = st.columns(3)
    w1.image(src_img, caption="Original", use_container_width=True)
    w2.image(warp_naive, caption="Naive (least-squares) warp",
             use_container_width=True)
    w3.image(warp_ransac, caption="RANSAC warp", use_container_width=True)
    st.caption(
        "Even a few outliers can pull the naive estimate far from the true "
        "transform. RANSAC recovers a near-correct warp by ignoring them."
    )

    st.markdown(
        """
        <div class="card">
            <h4>Why RANSAC is needed</h4>
            <p>Standard least-squares spreads its error across every match. Even a
            handful of grossly wrong correspondences can pull the estimated
            homography away from the true solution.</p>
            <p>RANSAC repeatedly samples a minimal subset of matches, fits a
            homography, and counts the matches that agree (inliers). The model
            with the most inliers wins — outliers are then ignored.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab 6: Blending Studio
# ---------------------------------------------------------------------------

with tab_blending:
    st.markdown('<div class="section-title">Blending Studio</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="caption">Two overlapping synthetic images are merged with the '
        "blending method of your choice. Watch the seam appear or disappear.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    method = c1.selectbox(
        "Blend method",
        ["Hard Cut", "Linear Blend", "Feathering",
         "Gaussian Mask Blend (simplified multi-band)"],
    )
    seam = c2.slider("Seam position (x)", 80, 400, 240, 1)
    c2.caption("Where image A ends and image B begins")
    overlap = c3.slider("Overlap width (px)", 20, 300, 160, 10)
    c3.caption("Controls the transition zone width for gradient methods")

    c4, c5 = st.columns(2)
    bright_diff = c4.slider("Brightness difference (B)", -80, 80, 30, 5)
    c4.caption("Simulates auto-exposure change between shots; high values reveal seam artifacts")
    blur = c5.slider("Blur / smoothing strength", 1, 60, 18, 1)
    c5.caption("Widens the transition zone for the Gaussian mask method")

    img_a, img_b, _ = make_two_overlapping_images(480, 320, overlap, bright_diff)
    mask = make_alpha_mask(img_a.shape[0], img_a.shape[1], seam, overlap, method, blur)
    blended = linear_blend(img_a, img_b, mask)

    cols = st.columns(3)
    cols[0].image(img_a, caption="Image A", use_container_width=True)
    cols[1].image(img_b, caption="Image B", use_container_width=True)
    cols[2].image(blended, caption=f"Blended ({method})", use_container_width=True)

    st.image((mask * 255).astype(np.uint8),
             caption="Alpha mask for image A (white = A, black = B)",
             use_container_width=True, clamp=True)

    st.markdown(
        """
        <div class="card">
            <h4>How each method behaves</h4>
            <ul>
                <li><b>Hard Cut:</b> picks A on one side and B on the other. Seams are obvious whenever exposure differs.</li>
                <li><b>Linear Blend:</b> averages pixels in the overlap region. Smooth, but can blur fine detail.</li>
                <li><b>Feathering:</b> uses a smooth cosine ramp instead of a linear ramp. Transitions are gentler.</li>
                <li><b>Gaussian Mask Blend:</b> a Gaussian-blurred alpha mask smooths the hard boundary. This approximates the idea behind multi-band blending but does NOT use Laplacian pyramids. Real multi-band blending decomposes images into frequency bands and blends each band at a different width. This version is a simplified approximation.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab 7: Failure Mode Gallery
# ---------------------------------------------------------------------------

with tab_failures:
    st.markdown('<div class="section-title">Failure Mode Gallery</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="caption">A tour of the situations where panorama stitching '
        "with a single homography breaks down.</div>",
        unsafe_allow_html=True,
    )

    failures = [
        ("Parallax", "#fb7185",
         "Camera translates instead of rotating about its optical center.",
         "Near objects appear shifted relative to far objects between the two views.",
         "A single homography assumes one global mapping; depth-dependent shifts cannot be expressed by one matrix.",
         "Rotate from a fixed point (tripod head), or use multi-plane / 3D methods."),
        ("Rolling shutter", "#f59e0b",
         "CMOS sensors expose rows over time while the camera moves.",
         "Vertical lines bend; the image looks sheared during fast motion.",
         "The two halves of the image were taken from slightly different camera poses, so no single homography fits.",
         "Use global shutter cameras or apply a rolling-shutter correction model."),
        ("Low texture", "#facc15",
         "Plain walls, sky, snow — regions without distinctive structure.",
         "Very few keypoints are found; matches are weak or absent.",
         "Without correspondences, the homography is unconstrained in those areas.",
         "Add textured regions to the overlap, or fall back to direct (pixel-based) alignment."),
        ("Repeated patterns", "#34d399",
         "Bricks, tiles, windows — visually identical features at different positions.",
         "Many descriptors look the same, so matches snap to the wrong copy.",
         "Wrong matches all look geometrically consistent in small groups, fooling RANSAC.",
         "Use spatial verification, larger image context, or guided matching."),
        ("Moving objects", "#22d3ee",
         "People, cars, leaves moving between the two shots.",
         "The same physical object is at different image positions in each shot.",
         "Matches on movers contradict the global homography; if they dominate, the fit drifts.",
         "Detect and remove movers, or use seam-aware blending that hides them."),
        ("Exposure differences", "#a78bfa",
         "Auto-exposure changed between shots, or lighting differs.",
         "A visible vertical brightness step appears at the seam.",
         "Geometry is fine but pixel intensities do not match across the seam.",
         "Use exposure compensation and multi-band blending."),
        ("Lens distortion", "#f472b6",
         "Wide-angle or fisheye lenses bend straight lines near the edges.",
         "Edges of the image curve; alignment is good in the center but bad on the sides.",
         "A homography is a global linear-rational map; it cannot undo radial distortion.",
         "Calibrate the camera and undistort each image before stitching."),
    ]

    cols = st.columns(2)
    for i, (name, color, cause, looks, why, fix) in enumerate(failures):
        with cols[i % 2]:
            st.markdown(
                f"""
                <div class="fail-card" style="background: linear-gradient(135deg, {color}, #fde68a);">
                    <h4>{name}</h4>
                    <p><b>Cause:</b> {cause}</p>
                    <p><b>What it looks like:</b> {looks}</p>
                    <p><b>Why homography struggles:</b> {why}</p>
                    <p><b>Possible fix:</b> {fix}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="card">
            <h4>When NOT to use a single homography</h4>
            <ul>
                <li>Strong camera translation (large baseline)</li>
                <li>Non-planar scenes with close objects (deep parallax)</li>
                <li>Moving objects that dominate the overlap region</li>
                <li>Severe rolling shutter caused by fast camera motion</li>
                <li>Wide-angle or fisheye lens distortion without calibration</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab 8: Full Stitching Demo
# ---------------------------------------------------------------------------

with tab_full:
    st.markdown('<div class="section-title">Full Stitching Demo</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="caption">The complete pipeline end to end: generate two '
        "overlapping views, detect features, match, RANSAC, warp and blend.</div>",
        unsafe_allow_html=True,
    )

    r1c1, r1c2, r1c3 = st.columns(3)
    overlap_amount = r1c1.slider("Overlap amount (px)", 80, 300, 160, 10,
                                 key="full_overlap")
    rotation_deg = r1c2.slider("Rotation (deg)", -30.0, 30.0, 12.0, 1.0,
                               key="full_rotation")
    bright_diff_full = r1c3.slider("Brightness difference", -60, 60, 20, 5,
                                   key="full_bright")

    r2c1, r2c2, r2c3 = st.columns(3)
    ratio_thresh = r2c1.slider("Lowe ratio threshold", 0.5, 0.95, 0.75, 0.01,
                               key="full_ratio")
    ransac_thresh = r2c2.slider("RANSAC threshold (px)", 1.0, 20.0, 5.0, 0.5,
                                key="full_ransac_thr")
    blend_method = r2c3.selectbox(
        "Blend method",
        ["Hard Cut", "Linear Blend", "Feathering", "Gaussian Mask Blend"],
        index=2,
        key="full_blend_method",
    )

    # 1) Generate the two views.
    full_a = make_synthetic_scene(480, 320)
    h_full = build_homography(
        rotation_deg=rotation_deg,
        scale=1.0,
        tx=overlap_amount * 0.3,
        ty=0.0,
        shear=0.0,
        p_x=0.0,
        p_y=0.0,
        center=(full_a.shape[1] / 2, full_a.shape[0] / 2),
    )
    full_b = warp_with_homography(full_a, h_full)
    full_b = np.clip(full_b.astype(np.int32) + bright_diff_full,
                     0, 255).astype(np.uint8)

    # 2) Feature detection and 3) Lowe ratio.
    kp_fa, kp_fb, good_full, det_full = detect_and_match(
        full_a, full_b, 800, ratio_thresh
    )

    n_good = len(good_full)
    H_full = None
    inlier_mask_full = None
    n_in_full = 0
    if n_good >= 4:
        sp = np.float32([kp_fa[m.queryIdx].pt for m in good_full]).reshape(-1, 1, 2)
        dp = np.float32([kp_fb[m.trainIdx].pt for m in good_full]).reshape(-1, 1, 2)
        # 4) RANSAC: estimate B -> A so we can warp B into A's frame.
        H_full, inlier_mask_full = cv2.findHomography(
            dp, sp, cv2.RANSAC, ransac_thresh,
        )
        if inlier_mask_full is not None:
            n_in_full = int(inlier_mask_full.sum())

    if n_good < 4 or H_full is None:
        st.markdown(
            f'<div class="warn-card">Not enough matches ({n_good}). '
            "Try increasing overlap or reducing rotation.</div>",
            unsafe_allow_html=True,
        )
    else:
        # 5) Warp B onto A's canvas.
        h_img, w_img = full_a.shape[:2]
        canvas_w = w_img * 2
        warped_full_b = cv2.warpPerspective(
            full_b, H_full, (canvas_w, h_img),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(15, 23, 42),
        )
        # 6) Blend in the overlap region using the selected method.
        canvas_a = np.zeros((h_img, canvas_w, 3), dtype=np.uint8)
        canvas_a[:, :w_img] = full_a
        seam_full = w_img
        full_mask = make_alpha_mask(
            h_img, canvas_w, seam_full, overlap_amount, blend_method, 18,
        )
        full_panorama = linear_blend(canvas_a, warped_full_b, full_mask)

        kp_vis_a = draw_keypoints(full_a, kp_fa)
        matches_vis = draw_matches_with_inliers(
            full_a, kp_fa, full_b, kp_fb, good_full, inlier_mask_full,
            max_draw=60,
        )

        # Row 1
        row1 = st.columns(3)
        row1[0].image(full_a, caption="Image A", use_container_width=True)
        row1[1].image(full_b, caption="Image B", use_container_width=True)
        row1[2].image(kp_vis_a, caption=f"{det_full} keypoints on A",
                      use_container_width=True)

        # Row 2
        row2 = st.columns(2)
        row2[0].image(
            matches_vis,
            caption="Feature matches (green = RANSAC inlier, red = outlier)",
            use_container_width=True,
        )
        with row2[1]:
            st.markdown("**Estimated H (B → A)**")
            H_norm = H_full / H_full[2, 2]
            st.dataframe(
                pd.DataFrame(
                    {
                        "col 0": [f"{H_norm[i, 0]: .4f}" for i in range(3)],
                        "col 1": [f"{H_norm[i, 1]: .4f}" for i in range(3)],
                        "col 2": [f"{H_norm[i, 2]: .4f}" for i in range(3)],
                    },
                    index=["h0", "h1", "h2"],
                ),
                use_container_width=True,
            )

        # Row 3
        row3 = st.columns(3)
        row3[0].image(warped_full_b, caption="Warped B",
                      use_container_width=True)
        row3[1].image(full_panorama, caption="Final blended panorama",
                      use_container_width=True)
        row3[2].image((full_mask * 255).astype(np.uint8),
                      caption="Alpha mask (white = A, black = B)",
                      use_container_width=True, clamp=True)

        # Metrics
        mm = st.columns(5)
        mm[0].metric("Keypoints A", len(kp_fa))
        mm[1].metric("Keypoints B", len(kp_fb))
        mm[2].metric("Good matches", n_good)
        mm[3].metric("RANSAC inliers", n_in_full)
        mm[4].metric("Blend", blend_method)

        if n_in_full < 10:
            st.markdown(
                f'<div class="warn-card">Too few inliers ({n_in_full}). '
                "The homography may be unreliable. Try increasing overlap, "
                "reducing rotation, or raising the RANSAC threshold.</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="card">
            <h4>What to try</h4>
            <p>This tab runs the complete panorama stitching pipeline end to end.
            Adjust the sliders to see how each parameter affects the output.</p>
            <ul>
                <li>High rotation tests SIFT's rotation invariance.</li>
                <li>High brightness difference reveals blending artifacts.</li>
                <li>Low ratio threshold keeps only the most confident matches.</li>
                <li>Low RANSAC threshold is strict — more matches become outliers.</li>
                <li>High overlap gives RANSAC more inliers to work with.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Tab 9: Interactive Quiz
# ---------------------------------------------------------------------------

@dataclass
class QuizQuestion:
    text: str
    options: List[str]
    correct_index: int  # 0-based
    explanation: str


QUIZ: List[QuizQuestion] = [
    QuizQuestion(
        text='Q1. The "Pure Rotation" assumption is critical for panorama stitching '
             "because a homography matrix fundamentally assumes:",
        options=[
            "A) The camera has zero focal length.",
            "B) All objects in the scene are at an infinite distance from the camera.",
            "C) The camera rotates strictly about its optical center, eliminating depth-dependent shifts.",
            "D) The lens has no physical aperture.",
        ],
        correct_index=2,
        explanation="A homography can only perfectly model the transformation between "
                    "two views when the camera rotates purely about its optical center. "
                    "Any translation introduces parallax — depth-dependent shifts between "
                    "objects at different distances — which a single homography cannot "
                    "represent.",
    ),
    QuizQuestion(
        text="Q2. A standard Homography matrix is a 3x3 matrix. How many degrees of "
             "freedom does it actually possess for modeling geometric relationships?",
        options=[
            "A) 9, because there are 9 entries in the matrix.",
            "B) 8, because it is a homogeneous matrix where one element is used for scale.",
            "C) 4, because that is the minimum number of points needed to solve it.",
            "D) 6, to account for rotation and translation in 3D space.",
        ],
        correct_index=1,
        explanation="A 3x3 homography matrix has 9 values, but because it operates in "
                    "homogeneous coordinates the overall scale is irrelevant. We can "
                    "always divide all entries by one non-zero element, leaving exactly "
                    "8 independent degrees of freedom.",
    ),
    QuizQuestion(
        text='Q3. If a feature is described as being "Scale Invariant" in the SIFT '
             "algorithm, what does this actually mean for the stitching process?",
        options=[
            "A) The feature will always be the same size regardless of the camera's resolution.",
            "B) The feature will only be detected if the camera is exactly 5 meters away from the object.",
            "C) The algorithm can recognize the same physical point even if one photo is zoomed in or taken from a different distance than the other.",
            "D) The feature ignores the edges of the image to focus only on the center.",
        ],
        correct_index=2,
        explanation="SIFT builds a scale-space pyramid and detects keypoints at the scale "
                    "where they are most stable. The descriptor is computed relative to "
                    "that characteristic scale, so the same feature is recognized even "
                    "when the camera distance or zoom changes.",
    ),
    QuizQuestion(
        text="Q4. In the SIFT algorithm, how is the Difference of Gaussian (DoG) "
             "function mathematically defined to identify potential features at "
             "various scales?",
        options=[
            "A) D(x, y, sigma) = Laplacian of L(x, y, sigma)",
            "B) D(x, y, sigma) = L(x, y, sigma squared) - L(x, y, sigma)",
            "C) D(x, y, sigma) = L(x, y, k sigma) - L(x, y, sigma)",
            "D) D(x, y, sigma) = G(x, y, k sigma) / G(x, y, sigma)",
        ],
        correct_index=2,
        explanation="The DoG is computed by subtracting two consecutive Gaussian-blurred "
                    "versions of the image separated by a multiplicative scale factor k. "
                    "This approximates the scale-normalised Laplacian of Gaussian and "
                    "highlights stable blob-like structures.",
    ),
    QuizQuestion(
        text="Q5. In SIFT's scale-space extrema detection, why is it conceptually "
             "necessary to compare a candidate pixel to 18 neighbors in adjacent "
             "scales rather than just the 8 spatial neighbors at its own scale?",
        options=[
            "A) To account for the 8 degrees of freedom required by the homography matrix.",
            "B) To ensure the feature is localized accurately in terms of illumination invariance.",
            "C) To ensure the feature is a local maximum or minimum across zoom levels, making it repeatable even if the camera moves closer or farther away.",
            "D) To reduce the computational complexity of the K-d tree search from O(n squared) to O(n log n).",
        ],
        correct_index=2,
        explanation="A true scale-space extremum must be extreme in both position and "
                    "scale. Checking neighbors in the scale above and below confirms "
                    "that the point is stable across zoom levels.",
    ),
    QuizQuestion(
        text="Q6. Why is RANSAC essentially required for the homography estimation "
             "step in panorama stitching?",
        options=[
            "A) It speeds up the K-d tree feature matching process.",
            "B) It calculates the average position of all detected keypoints to find the center.",
            "C) A single outlier, meaning an incorrect match, can completely corrupt a standard mathematical alignment.",
            "D) It is the only way to mathematically solve a 3x3 matrix.",
        ],
        correct_index=2,
        explanation="Least-squares solvers minimize total error across all matches, so "
                    "incorrect correspondences can skew the homography. RANSAC repeatedly "
                    "fits models from small random subsets and chooses the model "
                    "supported by the most inliers.",
    ),
    QuizQuestion(
        text="Q7. A photographer attempts to stitch two aerial photos of a city taken "
             "from a moving drone. Despite high-quality feature matches, the "
             "skyscrapers in the final panorama appear ghosted or shifted relative "
             "to the street level. What is the most likely theoretical cause?",
        options=[
            "A) The drone moved faster than the SIFT algorithm could process orientation gradients.",
            "B) The drone's translation introduced parallax, which a homography cannot model because it assumes the camera only rotates about its optical center.",
            "C) The skyscrapers were too tall for the Difference of Gaussian filter to detect extrema.",
            "D) The radial distortion of the drone's lens made feature matching impossible.",
        ],
        correct_index=1,
        explanation="When the drone translates, objects at different depths shift by "
                    "different amounts in the image plane. A single homography cannot "
                    "align street-level and rooftop objects at the same time.",
    ),
    QuizQuestion(
        text="Q8. If feature matching identifies 100 pairs of points, but 5 of those "
             "pairs are completely wrong, why can't we just use the average of all "
             "100 points to align the images?",
        options=[
            "A) Averaging is mathematically impossible with a 3x3 matrix.",
            "B) A single incorrect match can completely corrupt the mathematical alignment, resulting in a distorted image.",
            "C) The SIFT algorithm only allows the use of 4 points at a time.",
            "D) Averaging would make the final panorama too blurry to see.",
        ],
        correct_index=1,
        explanation="Standard least-squares methods treat all matches equally. Grossly "
                    "incorrect correspondences pull the estimated homography away from "
                    "the true solution. RANSAC is used to identify and discard outliers.",
    ),
    QuizQuestion(
        text="Q9. Which image blending technique is specifically noted for smoothing "
             "brightness differences while keeping sharp details clear?",
        options=[
            "A) Linear Blending",
            "B) Feathering",
            "C) Multi-band Blending",
            "D) Alpha Compositing",
        ],
        correct_index=2,
        explanation="Multi-band blending decomposes images into frequency bands and "
                    "blends each band differently. Low-frequency brightness differences "
                    "are blended gradually, while high-frequency details are preserved.",
    ),
    QuizQuestion(
        text="Q10. When a camera translates rather than just rotates, what specific "
             "error occurs that a homography cannot mathematically model?",
        options=[
            "A) Radial distortion",
            "B) Parallax",
            "C) Ghosting",
            "D) Motion blur",
        ],
        correct_index=1,
        explanation="Parallax is the apparent relative displacement of objects at "
                    "different depths caused by camera position change. A homography "
                    "assumes pure rotation or a planar scene, so it cannot model "
                    "depth-dependent displacement.",
    ),
]


with tab_quiz:
    st.markdown('<div class="section-title">Interactive Quiz</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="caption">Ten questions covering homography theory, SIFT, '
        "RANSAC, blending and panorama failure modes. Answer each question and "
        "press <b>Check Answer</b> to see the explanation.</div>",
        unsafe_allow_html=True,
    )
    init_quiz_state()

    for i, q in enumerate(QUIZ):
        st.markdown('<div class="quiz-card">', unsafe_allow_html=True)
        st.markdown(f"**{q.text}**")
        choice = st.radio(
            label="Select your answer",
            options=list(range(len(q.options))),
            format_func=lambda idx, opts=q.options: opts[idx],
            key=f"q_radio_{i}",
            index=st.session_state.quiz_answers.get(i, 0),
            label_visibility="collapsed",
        )
        st.session_state.quiz_answers[i] = choice

        if st.button("Check Answer", key=f"q_check_{i}"):
            st.session_state.quiz_checked[i] = True

        if st.session_state.quiz_checked.get(i):
            picked = st.session_state.quiz_answers[i]
            if picked == q.correct_index:
                st.markdown(
                    f'<div class="quiz-correct"><b>Correct.</b> '
                    f"{q.explanation}</div>",
                    unsafe_allow_html=True,
                )
            else:
                correct_letter = q.options[q.correct_index]
                st.markdown(
                    f'<div class="quiz-wrong"><b>Not quite.</b> The correct answer is: '
                    f"{correct_letter}<br><br>{q.explanation}</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # Final score
    answered = [i for i in range(len(QUIZ)) if st.session_state.quiz_checked.get(i)]
    score = sum(
        1 for i in answered
        if st.session_state.quiz_answers.get(i) == QUIZ[i].correct_index
    )
    total = len(QUIZ)

    st.markdown('<div class="card"><h4>Your Score</h4>', unsafe_allow_html=True)
    st.progress(score / total if total else 0.0,
                text=f"{score} / {total} correct ({len(answered)} answered)")

    if len(answered) == total:
        if score == total:
            msg = "Perfect score. You have mastered panorama stitching theory."
        elif score >= total * 0.8:
            msg = "Excellent. You have a strong grasp of the material."
        elif score >= total * 0.5:
            msg = "Good effort. Review the explanations above to fill the gaps."
        else:
            msg = "Keep practicing. Re-read the explanations and try again."
        st.markdown(f"<p><b>{msg}</b></p>", unsafe_allow_html=True)

    st.button("Reset Quiz", on_click=reset_quiz)
    st.markdown("</div>", unsafe_allow_html=True)
