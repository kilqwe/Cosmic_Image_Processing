import cv2
import numpy as np
import os

# --- GLOBAL CONFIGURATION ---
FPS = 30
STATIC_DURATION = 2.0
TRANSITION_DURATION = 0.8
TARGET_SIZE = (1920, 1080)

# NIRCam Sequence
SEQUENCE_1 = [
    ("raw_enhancement_pipeline_nircam.png", "RAW RGB: Baseline field with 1/f noise"),
    ("source_detection_nircam.png", "SOURCE DETECTION: 181 stellar points mapped"),
    ("fft_enhancement_pipeline_nircam.png", "FFT FILTER: Suppressing horizontal artifacts"),
    ("clahe_enhancement_pipeline_nircam.png", "CLAHE BOOST: Revealing debris structures"),
    ("fft+clahe_enhancement_pipeline_nircam.png", "FINAL PIPELINE: Cleaned & Optimized Signal"),
    ("hough_circle_nircam_1.png", "HOUGH ANALYSIS: Automated parameter search"),
    ("hough_circle_nircam_3.jpg", "CIRCLE FIT: Radius extraction (0.92 arcsec)")
]

# MIRI Sequence (New)
SEQUENCE_2 = [
    ("raw_miri_enhancement_pipeline.png", "RAW MIRI: Wide field emission (Prog 1232)"),
    ("miri_source_detection.png", "SOURCE DETECTION: Mapping thermal point sources"),
    ("fft_miri_enhancement_pipeline.png", "FFT FILTER: Eliminating MIRI detector artifacts"),
    ("clahe_miri_enhancement_pipeline.png", "CLAHE BOOST: Enhancing cooler dust signatures"),
    ("fft+clahe_miri_enhancement_pipeline.png", "FINAL PIPELINE: Optimized thermal signal"),
    ("miri_hough_circle_1.png", "HOUGH ANALYSIS: Automated parametric search"),
    ("miri_hough_circle_3.png", "CIRCLE FIT: Dust ring radius extraction")
]

def draw_caption(img, text):
    h, w = img.shape[:2]
    overlay = img.copy()
    rect_h = int(h * 0.12)
    cv2.rectangle(overlay, (0, h - rect_h), (w, h), (0, 0, 0), -1)
    alpha = 0.6
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.4
    thickness = 2
    color = (255, 255, 255)
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h - (rect_h // 2) + (text_size[1] // 2)
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    return img

def resize_and_pad(img, target_size):
    target_w, target_h = target_size
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def generate_sequence_video(sequence, output_name):
    frames = []
    for path, caption in sequence:
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping.")
            continue
        img = cv2.imread(path)
        img = resize_and_pad(img, TARGET_SIZE)
        img = draw_caption(img, caption)
        frames.append(img)

    if not frames:
        print(f"No frames found for {output_name}. Aborting.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_name, fourcc, FPS, TARGET_SIZE)
    static_frames = int(STATIC_DURATION * FPS)
    transition_frames = int(TRANSITION_DURATION * FPS)

    for i in range(len(frames)):
        for _ in range(static_frames):
            out.write(frames[i])
        if i < len(frames) - 1:
            for t in range(transition_frames):
                alpha = t / float(transition_frames)
                blended = cv2.addWeighted(frames[i+1], alpha, frames[i], 1.0 - alpha, 0.0)
                out.write(blended)
    
    out.release()
    print(f"Success: {output_name} generated.")

if __name__ == "__main__":
    generate_sequence_video(SEQUENCE_1, "sn1987a_nircam_demo.mp4")
    generate_sequence_video(SEQUENCE_2, "sn1987a_miri_demo.mp4")