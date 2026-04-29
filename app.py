import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import feature
from skimage.transform import hough_circle, hough_circle_peaks
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image
import json
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JWST SN1987A — CV Pipeline",
    page_icon="🔭",
    layout="wide",
)

st.title("🔭 JWST SN1987A — Computer Vision Pipeline")
st.markdown(
    "Processing real **James Webb Space Telescope** data. "
    "Target: SN 1987A — an active supernova in the Large Magellanic Cloud."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select pipeline", [
    "Overview",
    "Enhancement",
    "Detection",
    "Results Summary",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.markdown("- Program 1232 — wide field")
st.sidebar.markdown("- Program 1726 — sub320 ring")
st.sidebar.markdown("- Filters: F115W, F277W, F444W")


# ── Helper: load image ────────────────────────────────────────────────────────
def load_img(path):
    p = Path(path)
    if p.exists():
        return Image.open(p)
    return None

def load_npy(path):
    p = Path(path)
    if p.exists():
        return np.load(p)
    return None


# ── Page: Overview ────────────────────────────────────────────────────────────
if page == "Overview":
    st.header("Project Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("What we built")
        st.markdown("""
        A complete computer vision pipeline on real JWST observations:

        **ENHANCEMENT PIPELINE**
        - ORB feature-based alignment across 3 filter bands
        - Wavelet decomposition + fusion (db4, level 4)
        - FFT power spectrum analysis + DC notch filter
        - CLAHE local contrast enhancement
        - False-color RGB composite (Lupton asinh stretch)

        **DETECTION PIPELINE**
        - Edge detection: Sobel, Canny, Laplacian of Gaussian
        - Morphological operations: erosion, dilation, opening, closing
        - Source detection: DAOStarFinder (181 sources)
        - Hough circle transform on SN ring
        - PCA decomposition across 6 filter bands
        """)

    with col2:
        st.subheader("Key result")
        rgb = load_img("Images/sn1987a_enhanced_final.png")
        if rgb:
            st.image(rgb, caption="SN 1987A — FFT + CLAHE enhanced RGB",
                     use_container_width=True)
        else:
            st.info("Run enhance.ipynb first to generate outputs")

    st.markdown("---")
    st.subheader("CV Concepts Covered")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frequency domain", "FFT + notch filter")
    c2.metric("Spatial filtering", "Sobel, Canny, LoG")
    c3.metric("Sources detected", "181")
    c4.metric("Ring radius", "0.92 arcsec")

    # ── Wide field imagery section ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Wide Field Imagery — Raw JWST Data")
    st.markdown(
        "False-color RGB composite from raw JWST NIRCam observations. "
        "Three infrared filters are mapped to visible colors: "
        "**F115W → blue** (1.15μm, hot stars), "
        "**F277W → green** (2.77μm, warm gas), "
        "**F444W → red** (4.44μm, warm dust). "
        "Thousands of stars from the Large Magellanic Cloud are visible, "
        "along with the SN1987A remnant and its surrounding nebular structure."
    )

    # Best single RGB — full width
    best_rgb = load_img("Images/sn1987a_rgb.png")
    if best_rgb:
        st.image(best_rgb,
                 caption="SN 1987A wide field — F115W/F277W/F444W (Lupton asinh stretch, 99th pct, Q=5)",
                 use_container_width=True)
    else:
        st.warning("sn1987a_rgb.png not found — run the wide field cell in enhance.ipynb")

    st.markdown("---")

    # Dynamic range comparison — 3 variants side by side
    st.subheader("Dynamic Range Comparison")
    st.markdown(
        "The same raw data stretched three different ways. "
        "The middle panel (99.0th percentile, Q=5) gives the best balance "
        "between star brightness and faint nebular structure."
    )
    comp_img = load_img("Images/sn1987a_comparison.png")
    if comp_img:
        st.image(comp_img,
                 caption="Three Lupton stretch variants — left: Q=8, centre: Q=5 (best), right: Q=3",
                 use_container_width=True)

    st.markdown("---")

    # Widefield zoom
    st.subheader("Ring Region — Zoomed")
    st.markdown(
        "The SN1987A circumstellar ring is visible as a bright compact "
        "source in the upper-right of the wide field. The yellow box "
        "marks the region containing the supernova remnant."
    )
    zoom_img = load_img("Images/sn1987a_widefield_zoom.png")
    if zoom_img:
        st.image(zoom_img,
                 caption="Wide field with zoom box (left) and ring region zoomed (right)",
                 use_container_width=True)

    st.markdown("---")

    # Individual filter bands
    st.subheader("Individual Filter Bands")
    st.markdown(
        "Each FITS file contains one filter band. "
        "The ring is most prominent in **F444W** (4.44μm) — "
        "warm dust heated by the expanding shockwave glows brightest "
        "in the infrared. F115W captures hot stellar populations, "
        "F277W shows warm molecular gas."
    )
    bands_img = load_img("Images/sn1987a_widefield_bands.png")
    if bands_img:
        st.image(bands_img,
                 caption="F115W / F277W / F444W — individual bands, ZScale stretch",
                 use_container_width=True)

    st.markdown("---")

    # Sub320 ring
    st.subheader("Sub320 Ring — Targeted Observation")
    st.markdown(
        "Program 1726 used a 320×320 pixel subarray pointed directly at "
        "the SN ring. The direct stack shows the ring in two wavelength "
        "regimes: **red = F444W** (warm dust, 4.44μm) and "
        "**cyan = F150W/F200W** (cooler gas, 1.5–2.0μm). "
        "The offset between the two rings shows the bands are not "
        "natively aligned — motivating the WCS alignment step."
    )
    ring_img = load_img("Images/sn1987a_direct.png")
    if ring_img:
        st.image(ring_img,
                 caption="Sub320 direct stack — ring visible in two wavelength regimes",
                 use_container_width=True)


# ── Page: Enhancement ─────────────────────────────────────────────────────────
elif page == "Enhancement":
    st.header("Enhancement Pipeline")

    st.subheader("1. ORB Alignment + Wavelet Merge")
    st.markdown("""
    Three NIRCam filter bands (F115W, F277W, F444W) are aligned using ORB
    keypoint matching. The homography matrix maps F115W and F277W onto the
    F444W reference frame. Aligned bands are fused using db4 wavelet
    decomposition — at each subband the maximum coefficient is kept,
    preserving the sharpest detail from whichever filter captured it best.
    """)
    align_img = load_img("Images/alignment_result.png")
    if align_img:
        st.image(align_img, caption="ORB alignment result — 4 panels",
                 use_container_width=True)

    st.markdown("---")
    st.subheader("2. FFT Power Spectrum Analysis")
    st.markdown("""
    The 2D FFT of F277W shows a smooth power spectrum with no bright lines —
    confirming the wide-field mosaic has no strong periodic noise.
    A DC-only notch filter was applied. The sub320 F150W data showed
    horizontal banding (1/f noise), removed with a Gaussian notch filter.
    """)
    col1, col2 = st.columns(2)
    fft_img = load_img("Images/fft_widefield_gentle.png")
    fft_sub = load_img("Images/fft_sub320_banding.png")
    if fft_img:
        col1.image(fft_img, caption="Wide field — DC notch",
                   use_container_width=True)
    if fft_sub:
        col2.image(fft_sub, caption="Sub320 — banding removal",
                   use_container_width=True)

    st.markdown("---")
    st.subheader("3. CLAHE + Final RGB")
    st.markdown("""
    CLAHE (Contrast Limited Adaptive Histogram Equalisation) boosts local
    contrast in 8×8 pixel tiles. Applied post-FFT, it reveals faint structure
    near bright sources — the SN ring boundary becomes clearly resolvable.
    """)
    enh_img = load_img("Images/sn1987a_enhanced_final.png")
    if enh_img:
        st.image(enh_img, caption="Original vs FFT+CLAHE enhanced",
                 use_container_width=True)

    st.markdown("---")
    st.subheader("4. Metrics")

    metrics_path = Path("enhancement_metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        col1, col2, col3 = st.columns(3)
        col1.metric("PSNR", f"{m['psnr_db']:.2f} dB")
        col2.metric("SSIM", f"{m['ssim']:.4f}")
        col3.metric("Pipeline", m["pipeline"])
        st.info(
            "PSNR and SSIM measure similarity between original and enhanced. "
            "Low values are expected for CLAHE — the technique intentionally "
            "redistributes the histogram. The visual improvement (ring structure "
            "clearly resolvable) is the primary quality metric."
        )
    else:
        st.warning("Run enhance.ipynb Cell 8 to generate metrics")


# ── Page: Detection ───────────────────────────────────────────────────────────
elif page == "Detection":
    st.header("Detection Pipeline")

    st.subheader("1. Edge Detection")
    st.markdown("""
    Four edge detectors applied to the enhanced F444W band:
    - **Sobel**: gradient magnitude — smooth, continuous edges
    - **Canny σ=1.0**: fine edges — detects individual ring hotspots
    - **Canny σ=3.0**: coarse edges — clean ring boundary
    - **LoG σ=2.0**: Laplacian of Gaussian — zero-crossing detection
    """)
    edge_img = load_img("Images/edge_detection.png")
    if edge_img:
        st.image(edge_img, caption="Edge detection comparison",
                 use_container_width=True)

    st.markdown("---")
    st.subheader("2. Morphological Operations")
    st.markdown("""
    Binary morphological operations on the Canny edge map:
    - **Erosion**: removes thin noise connections between sources
    - **Dilation**: expands edge boundaries, connects nearby fragments
    - **Opening**: erosion then dilation — removes small noise blobs
    - **Closing**: dilation then erosion — fills gaps in ring contour
    """)
    morph_img = load_img("Images/morphological_ops.png")
    if morph_img:
        st.image(morph_img, caption="Morphological operations",
                 use_container_width=True)

    st.markdown("---")
    st.subheader("3. Source Detection")
    st.markdown("""
    DAOStarFinder detected **181 sources** in the F444W wide-field band
    at 5σ above background. Each source centroid was converted to RA/Dec
    using the WCS header and saved to source_catalog.csv.
    The SN1987A ring itself appears as multiple detections along its
    bright hotspot boundary.
    """)
    src_img = load_img("Images/source_detection.png")
    if src_img:
        st.image(src_img, caption="181 sources detected — cyan circles",
                 use_container_width=True)

    catalog = Path("source_catalog.csv")
    if catalog.exists():
        import pandas as pd
        df = pd.read_csv(catalog)
        st.markdown(f"**Source catalog** — {len(df)} sources")
        st.dataframe(df[["id", "xcentroid", "ycentroid",
                          "peak", "flux"]].head(10),
                     use_container_width=True)

    st.markdown("---")
    st.subheader("4. Hough Circle Transform")
    st.markdown("""
    The Hough circle transform votes for circles of varying radii in the
    Canny edge map. Applied to a cropped region around the ring to avoid
    false positives from field stars.
    """)
    hough_img = load_img("Images/hough_circle.png")
    if hough_img:
        st.image(hough_img, caption="Hough circle detection",
                 use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Detected radius", "54 px = 0.92\"")
    col2.metric("Literature value", "0.85 arcsec")
    col3.metric("Difference", "0.07 arcsec (8%)")
    st.success(
        "The 8% difference is consistent with the expanding shock front — "
        "SN1987A's ring is actively growing and JWST data is more recent "
        "than most published measurements."
    )

    st.markdown("---")
    st.subheader("5. PCA Decomposition")
    st.markdown("""
    All available filter bands stacked into a data cube and decomposed
    with PCA. Each principal component separates a different physical
    emission process:
    - **PC1 (79.7%)**: overall brightness — dominated by stellar continuum
    - **PC2 (10.7%)**: ring-specific emission — SN ring isolated from stars
    - **PC3 (9.6%)**: spectral contrast — ring has negative loading,
      spectrally distinct from background population
    """)
    pca_img = load_img("Images/pca_components.png")
    if pca_img:
        st.image(pca_img, caption="PCA decomposition — 6 components",
                 use_container_width=True)

    pca_path = Path("pca_results.json")
    if pca_path.exists():
        with open(pca_path) as f:
            pca_data = json.load(f)
        st.markdown("**Explained variance per component:**")
        cols = st.columns(len(pca_data["explained_variance_pct"]))
        for i, (col, var) in enumerate(
                zip(cols, pca_data["explained_variance_pct"])):
            col.metric(f"PC{i+1}", f"{var:.1f}%")


# ── Page: Results Summary ─────────────────────────────────────────────────────
elif page == "Results Summary":
    st.header("Results Summary")

    st.subheader("Enhancement")
    c1, c2, c3 = st.columns(3)
    c1.metric("Alignment", "ORB + Homography")
    c2.metric("Fusion", "Wavelet db4 level 4")
    c3.metric("Enhancement", "FFT + CLAHE")

    metrics_path = Path("enhancement_metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        c1, c2 = st.columns(2)
        c1.metric("PSNR", f"{m['psnr_db']:.2f} dB")
        c2.metric("SSIM", f"{m['ssim']:.4f}")

    st.markdown("---")
    st.subheader("Detection")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Edge detectors", "4 methods")
    c2.metric("Sources found", "181")
    c3.metric("Ring radius", "0.92 arcsec")
    c4.metric("Literature", "0.85 arcsec")

    pca_path = Path("pca_results.json")
    if pca_path.exists():
        with open(pca_path) as f:
            p = json.load(f)
        st.metric("PCA variance (PC1+PC2+PC3)",
                  f"{sum(p['explained_variance_pct'][:3]):.1f}%")

    st.markdown("---")
    st.subheader("Key findings")
    st.markdown("""
    1. ORB alignment works well on star-dense wide fields but fails on
       sub320 subarrays — WCS reprojection is the correct method for
       compact targeted observations.
    2. FFT power spectrum confirmed the wide-field mosaic is clean — no
       periodic noise. Sub320 data had real 1/f banding which was removed.
    3. CLAHE revealed the SN ring boundary which was unresolvable in the
       raw stretched image.
    4. Hough circle measured the ring at **0.92 arcsec** — 8% above the
       published value, consistent with ring expansion since publication.
    5. PCA PC2 isolates the ring from field stars — demonstrating the
       ring has a unique spectral signature across JWST filter bands.
    """)

    st.markdown("---")
    st.subheader("Output files")
    output_files = [
        "Images/sn1987a_rgb.png",
        "Images/sn1987a_comparison.png",
        "Images/sn1987a_widefield_bands.png",
        "Images/sn1987a_widefield_zoom.png",
        "Images/sn1987a_direct.png",
        "Images/sn1987a_enhanced_final.png",
        "Images/alignment_result.png",
        "Images/fft_widefield_gentle.png",
        "Images/fft_sub320_banding.png",
        "Images/edge_detection.png",
        "Images/morphological_ops.png",
        "Images/source_detection.png",
        "Images/hough_circle.png",
        "Images/pca_components.png",
        "source_catalog.csv",
        "enhancement_metrics.json",
        "pca_results.json",
        "Images/sn1987a_widefield_bands.png",
    ]
    for fname in output_files:
        exists = Path(fname).exists()
        st.markdown(f"{'Created' if exists else 'Not Created'} `{fname}`")