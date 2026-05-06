# ── app.py — Streamlit demo ───────────────────────────────────────────────────
# pip install streamlit
# Run with: streamlit run app.py

import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image
import json

st.set_page_config(
    page_title="JWST SN1987A — CV Pipeline",
    page_icon="🔭",
    layout="wide",
)

st.title("🔭 JWST SN1987A — Computer Vision Pipeline")
st.markdown(
    "Processing real **James Webb Space Telescope** data. "
    "Target: **SN 1987A** — an active supernova in the Large Magellanic Cloud. "
    "Two instruments, two wavelength regimes, one unified CV pipeline."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select page", [
    "Overview",
    "NIRCam Pipeline",
    "MIRI Pipeline",
    "Results Summary",
])

st.sidebar.markdown("---")
st.sidebar.markdown("**NIRCam dataset**")
st.sidebar.markdown("- Program 1232 + 1726")
st.sidebar.markdown("- F115W, F277W, F444W")
st.sidebar.markdown("- 1.15 – 4.44 μm")
st.sidebar.markdown("---")
st.sidebar.markdown("**MIRI dataset**")
st.sidebar.markdown("- Program 1232")
st.sidebar.markdown("- F560W, F1000W, F2550W")
st.sidebar.markdown("- 5.6 – 25.5 μm")


def load_img(path):
    p = Path(path)
    if p.exists():
        return Image.open(p)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.header("Project Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("What we built")
        st.markdown("""
        A complete computer vision pipeline on real JWST observations
        across **two instruments** covering different physics:

        **NIRCam** (1.15 – 4.44 μm) — stellar populations, ring structure
        **MIRI** (5.6 – 25.5 μm) — warm dust, ejecta, circumstellar gas

        **Enhancement pipeline**
        - ORB feature-based alignment + wavelet fusion
        - FFT power spectrum analysis + notch filtering
        - CLAHE local contrast enhancement
        - Lupton asinh false-color compositing

        **Detection pipeline**
        - Edge detection: Sobel, Canny, Laplacian of Gaussian
        - Morphological operations: erosion, dilation, opening, closing
        - Source detection: DAOStarFinder
        - Hough circle transform
        - PCA decomposition across all bands
        """)

    with col2:
        st.subheader("NIRCam — wide field RGB")
        img = load_img("Images/sn1987a_rgb.png")
        if img:
            st.image(img, caption="F115W/F277W/F444W — Lupton stretch",
                     width='stretch')
        else:
            st.info("Run enhance.ipynb to generate images")

    st.markdown("---")

    # Instrument comparison
    st.subheader("Two instruments — same target")
    st.markdown(
        "NIRCam and MIRI observe the same field but reveal completely different "
        "physical components. NIRCam captures hot stars and the compact ring. "
        "MIRI reveals warm dust emission from the ejecta and surrounding nebula "
        "that is invisible in the near-infrared."
    )

    col1, col2 = st.columns(2)
    with col1:
        nircam = load_img("Images/sn1987a_enhanced_final.png")
        if nircam:
            st.image(nircam, caption="NIRCam — FFT + CLAHE enhanced (1–5 μm)",
                     width='stretch')
    with col2:
        miri = load_img("Images/miri_enhanced_final.png")
        if miri:
            st.image(miri, caption="MIRI — FFT + CLAHE enhanced (5–25 μm)",
                     width='stretch')

    st.markdown("---")
    st.subheader("CV Concepts Covered")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Instruments", "NIRCam + MIRI")
    c2.metric("Filter bands", "6 total")
    c3.metric("Sources (NIRCam)", "181")
    c4.metric("Ring radius", "0.92 arcsec")
    c5.metric("PCA components", "6")

    st.markdown("---")
    st.subheader("Wide field gallery")
    comp = load_img("Images/sn1987a_comparison.png")
    if comp:
        st.image(comp, caption="Dynamic range comparison — 3 Lupton stretch variants",
                 width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: NIRCam PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "NIRCam Pipeline":
    st.header("NIRCam Pipeline — 1.15 to 4.44 μm")
    st.markdown(
        "Near-infrared observations from Programs 1232 and 1726. "
        "Three filters: F115W (blue, hot stars), F277W (green, warm gas), "
        "F444W (red, warm dust). Wide-field mosaic + sub320 ring subarray."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Enhancement", "Edge Detection", "Source Detection & Hough", "PCA"
    ])

    with tab1:
        st.subheader("ORB Alignment + Wavelet Fusion")
        st.markdown("""
        Three NIRCam bands aligned using ORB keypoint matching. Homography
        estimated with RANSAC. Bands fused with db4 wavelet decomposition —
        maximum coefficient at each subband preserves sharpest detail.
        """)
        align = load_img("Images/alignment_result.png")
        if align:
            st.image(align, caption="ORB alignment — 4 panel result",
                     width='stretch')

        st.markdown("---")
        st.subheader("FFT Power Spectrum + Filtering")
        col1, col2 = st.columns(2)
        f1 = load_img("Images/fft_widefield_gentle.png")
        f2 = load_img("Images/fft_sub320_banding.png")
        if f1:
            col1.image(f1, caption="Wide field — DC notch (clean data)",
                       width='stretch')
        if f2:
            col2.image(f2, caption="Sub320 — Gaussian notch (real 1/f banding)",
                       width='stretch')

        st.markdown("---")
        st.subheader("CLAHE + Final Enhanced RGB")
        enh = load_img("Images/sn1987a_enhanced_final.png")
        if enh:
            st.image(enh, caption="Original vs FFT + CLAHE enhanced",
                     width='stretch')

        metrics_path = Path("enhancement_metrics.json")
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            c1, c2 = st.columns(2)
            c1.metric("PSNR", f"{m['psnr_db']:.2f} dB")
            c2.metric("SSIM", f"{m['ssim']:.4f}")
            st.info(
                "Low PSNR/SSIM are expected for CLAHE — the technique "
                "intentionally redistributes the histogram. Visual ring "
                "detectability is the primary quality metric."
            )

    with tab2:
        st.subheader("Edge Detection Comparison")
        st.markdown("""
        Four edge detectors applied to the enhanced F444W band:
        - **Sobel**: gradient magnitude — smooth continuous edges
        - **Canny σ=1.0**: fine edges — individual ring hotspots
        - **Canny σ=3.0**: coarse edges — clean ring perimeter
        - **LoG σ=2.0**: Laplacian of Gaussian — zero-crossing detection
        """)
        edge = load_img("Images/edge_detection_nircam.png")
        if edge:
            st.image(edge, caption="Edge detection comparison — 6 panels",
                     width='stretch')

        st.markdown("---")
        st.subheader("Morphological Operations")
        st.markdown("""
        Binary morphological operations on the Canny edge map:
        erosion removes thin noise, dilation expands boundaries,
        opening removes small blobs, closing fills ring contour gaps.
        """)
        morph = load_img("Images/morphological_ops_nircam.png")
        if morph:
            st.image(morph, caption="Morphological operations — 6 panels",
                     width='stretch')

    with tab3:
        st.subheader("Source Detection — DAOStarFinder")
        st.markdown(
            "181 sources detected in F444W at 5σ above background. "
            "Centroids converted to RA/Dec via WCS. The ring appears "
            "as multiple detections along its bright hotspot boundary."
        )
        src = load_img("Images/source_detection_nircam.png")
        if src:
            st.image(src, caption="181 sources — cyan circles",
                     width='stretch')

        catalog = Path("source_catalog.csv")
        if catalog.exists():
            import pandas as pd
            df = pd.read_csv(catalog)
            st.markdown(f"**Source catalog** — {len(df)} sources")
            st.dataframe(df[["id","xcentroid","ycentroid","peak","flux"]].head(10),
                         width='stretch')

        st.markdown("---")
        st.subheader("Hough Circle Transform")
        st.markdown("""
        Canny edge detection on a 160×160px crop centred on the ring.
        Hough circle transform across radii 15–55px. Top detection
        converted from pixels to arcseconds via WCS pixel scale.
        """)
        hough = load_img("Images/hough_circle_nircam.png")
        if hough:
            st.image(hough, caption="Hough circle detection — focused crop",
                     width='stretch')

        c1, c2, c3 = st.columns(3)
        c1.metric("Detected radius", "0.92 arcsec")
        c2.metric("Literature value", "0.85 arcsec")
        c3.metric("Difference", "8%")
        st.success(
            "8% difference is consistent with active ring expansion — "
            "SN1987A's shock front grows at ~3,500 km/s and JWST data "
            "post-dates most published measurements."
        )

    with tab4:
        st.subheader("PCA Decomposition")
        st.markdown("""
        All filter bands stacked into a pixel × band matrix. PCA applied
        with 6 components. Each component separates a different physical
        emission process:
        - **PC1 (79.7%)**: overall brightness — stellar continuum
        - **PC2 (10.7%)**: ring-specific emission — isolated from stars
        - **PC3 (9.6%)**: spectral contrast — ring has negative loading
        """)
        pca = load_img("Images/pca_components_nircam.png")
        if pca:
            st.image(pca, caption="PCA decomposition — 6 components",
                     width='stretch')

        pca_path = Path("pca_results.json")
        if pca_path.exists():
            with open(pca_path) as f:
                pd = json.load(f)
            cols = st.columns(len(pd["explained_variance_pct"]))
            for i, (col, var) in enumerate(
                    zip(cols, pd["explained_variance_pct"])):
                col.metric(f"PC{i+1}", f"{var:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MIRI PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "MIRI Pipeline":
    st.header("MIRI Pipeline — 5.6 to 25.5 μm")
    st.markdown(
        "Mid-infrared observations from Program 1232 using MIRI. "
        "Three filters: F560W (5.6μm), F1000W (10μm), F2550W (25.5μm). "
        "At these wavelengths MIRI captures warm dust emission, "
        "polycyclic aromatic hydrocarbons (PAH), and silicate features "
        "that are completely invisible to NIRCam."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Enhancement", "Edge Detection", "Source Detection", "PCA"
    ])

    with tab1:
        st.subheader("Raw MIRI + Enhanced RGB")
        st.markdown("""
        MIRI filters span a 5× larger wavelength range than NIRCam,
        meaning each filter captures fundamentally different dust and
        gas emission processes. F560W shows PAH emission, F1000W
        captures silicate absorption, F2550W traces the warmest dust.
        """)
        miri_raw = load_img("Images/miri_raw_enhanced.png")
        if miri_raw:
            st.image(miri_raw,
                     caption="Raw MIRI F1000W (left) vs FFT+CLAHE enhanced RGB (right)",
                     width='stretch')
        else:
            st.info("Save your MIRI enhanced comparison as Images/miri_raw_enhanced.png")

        st.markdown("---")
        st.subheader("FFT Power Spectrum")
        fft_miri = load_img("Images/miri_fft_power.png")
        if fft_miri:
            st.image(fft_miri,
                     caption="F1000W FFT power spectrum — smooth, no periodic noise",
                     width='stretch')
        else:
            st.info("Save as Images/miri_fft_power.png")

        st.markdown("---")
        st.subheader("Enhancement Pipeline")
        st.markdown("""
        Same FFT + CLAHE pipeline applied to MIRI bands. The mid-IR
        data has a smooth power spectrum confirming no periodic noise.
        DC-only notch filter applied. CLAHE reveals the filamentary
        nebular structure surrounding the compact remnant.
        """)
        miri_enh = load_img("Images/miri_enhancement_pipeline.png")
        if miri_enh:
            st.image(miri_enh,
                     caption="Enhancement pipeline — Raw / FFT / CLAHE / FFT+CLAHE",
                     width='stretch')
        else:
            st.info("Save as Images/miri_enhancement_pipeline.png")

        col1, col2 = st.columns(2)
        f1 = load_img("Images/miri_fft_banding.png")
        f2 = load_img("Images/miri_fft_gentle.png")
        if f1:
            col1.image(f1, caption="MIRI FFT — banding filter",
                       width='stretch')
        if f2:
            col2.image(f2, caption="MIRI FFT — DC notch",
                       width='stretch')

    with tab2:
        st.subheader("Edge Detection on MIRI")
        st.markdown("""
        The same four edge detectors applied to MIRI F2550W.
        The compact SN1987A remnant dominates as a bright circular
        source — clearly separated from the extended nebular emission
        by all detectors. The Laplacian of Gaussian is particularly
        effective at distinguishing the remnant boundary from the
        smooth dust background.
        """)
        edge_miri = load_img("Images/miri_edge_detection.png")
        if edge_miri:
            st.image(edge_miri,
                     caption="Edge detection on MIRI F2550W — 6 panels",
                     width='stretch')
        else:
            st.info("Save as Images/miri_edge_detection.png")

        st.markdown("---")
        st.subheader("Morphological Operations")
        morph_miri = load_img("Images/miri_morphological_ops.png")
        if morph_miri:
            st.image(morph_miri,
                     caption="Morphological operations on MIRI edge map",
                     width='stretch')
        else:
            st.info("Save as Images/miri_morphological_ops.png")

    with tab3:
        st.subheader("Source Detection — MIRI F2550W")
        st.markdown("""
        DAOStarFinder applied to MIRI F2550W. Only 3 sources detected —
        expected, because at 25.5μm the field is dominated by diffuse
        extended emission. Point-like sources are rare at this wavelength.
        The compact SN1987A remnant is the dominant detected source.
        """)
        src_miri = load_img("Images/miri_source_detection.png")
        if src_miri:
            st.image(src_miri,
                     caption="3 sources detected in F2550W",
                     width='stretch')
        else:
            st.info("Save as Images/miri_source_detection.png")

        c1, c2, c3 = st.columns(3)
        c1.metric("Sources (F2550W)", "3")
        c2.metric("Sources (NIRCam F444W)", "181")
        c3.metric("Reason", "Diffuse mid-IR emission")
        st.info(
            "The large difference in source counts between NIRCam (181) "
            "and MIRI (3) is physically meaningful — stars are bright in "
            "the near-IR but faint at 25μm. Extended dust emission fills "
            "the field, suppressing point-source detection."
        )

        st.markdown("---")
        st.subheader("Hough Circle — MIRI")
        hough_miri = load_img("Images/miri_hough_circle.png")
        if hough_miri:
            st.image(hough_miri,
                     caption="Hough circle on MIRI F2550W remnant",
                     width='stretch')
        else:
            st.info("Save as Images/miri_hough_circle.png")

    with tab4:
        st.subheader("PCA Decomposition — MIRI")
        st.markdown("""
        PCA applied to the 3 MIRI bands. With only 3 bands, PCA gives
        3 components. PC1 captures the dominant dust continuum emission.
        PC2 separates the compact remnant from the extended nebula —
        demonstrating a clear spectral difference between the supernova
        ejecta and the surrounding LMC interstellar medium.
        """)
        pca_miri = load_img("Images/miri_pca_components.png")
        if pca_miri:
            st.image(pca_miri,
                     caption="PCA decomposition — MIRI 3-band",
                     width='stretch')
        else:
            st.info("Save as Images/miri_pca_components.png")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Results Summary":
    st.header("Results Summary")

    st.markdown("---")
    st.subheader("Collated results videos")
    st.markdown(
        "Provide local MP4 file paths or URLs for the NIRCam and MIRI collated results videos. "
        "If the file exists, the video will be displayed below."
    )
    nircam_video_path = st.text_input(
        "NIRCam video path",
        value="Images/sn1987a_nircam_demo.mp4",
        help="Set the local MP4 path or URL for the NIRCam collated results video."
    )
    miri_video_path = st.text_input(
        "MIRI video path",
        value="Images/sn1987a_miri_demo.mp4",
        help="Set the local MP4 path or URL for the MIRI collated results video."
    )

    st.markdown("**NIRCam collated results video**")
    st.write(f"Path: {nircam_video_path}, Exists: {Path(nircam_video_path).exists()}")
    st.write(f"Absolute path: {Path(nircam_video_path).resolve()}")
    if nircam_video_path:
        if Path(nircam_video_path).exists() or nircam_video_path.startswith(("http://", "https://")):
            st.video(str(Path(nircam_video_path).resolve()))
        else:
            st.info("Set a valid NIRCam MP4 path or URL once available.")

    st.markdown("**MIRI collated results video**")
    st.write(f"Path: {miri_video_path}, Exists: {Path(miri_video_path).exists()}")
    st.write(f"Absolute path: {Path(miri_video_path).resolve()}")
    if miri_video_path:
        if Path(miri_video_path).exists() or miri_video_path.startswith(("http://", "https://")):
            st.video(str(Path(miri_video_path).resolve()))
        else:
            st.info("Set a valid MIRI MP4 path or URL once available.")

    st.markdown("---")
    st.subheader("Enhancement pipeline")
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
    st.subheader("Detection pipeline")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NIRCam sources", "181")
    c2.metric("MIRI sources", "3")
    c3.metric("Ring radius", "0.92 arcsec")
    c4.metric("Literature", "0.85 arcsec")

    pca_path = Path("pca_results.json")
    if pca_path.exists():
        with open(pca_path) as f:
            p = json.load(f)
        st.metric("NIRCam PCA variance (PC1+PC2+PC3)",
                  f"{sum(p['explained_variance_pct'][:3]):.1f}%")

    st.markdown("---")
    st.subheader("Key findings")
    st.markdown("""
    1. **ORB alignment is dataset-dependent.** Works on star-dense NIRCam
       wide fields but fails on sub320 subarrays due to the detector gap
       artifact. WCS reprojection is required for compact observations.

    2. **FFT confirmed clean calibration.** Both NIRCam and MIRI wide-field
       mosaics show smooth power spectra — no periodic noise. Sub320 NIRCam
       data has real 1/f banding requiring explicit frequency-domain filtering.

    3. **CLAHE reveals ring structure.** The circumstellar ring boundary
       is unresolvable in raw stretched data. Post-CLAHE it becomes
       clearly circular, enabling the Hough circle detection.

    4. **Hough circle confirms ring expansion.** Measured radius 0.92 arcsec
       vs 0.85 arcsec published — consistent with the actively expanding
       shock front at ~3,500 km/s.

    5. **NIRCam and MIRI trace different physics.** NIRCam PC2 isolates
       the compact ring, while MIRI PC2 separates ejecta from the LMC ISM.
       Together they provide a complete multi-wavelength picture of the
       supernova remnant.

    6. **Source count reflects wavelength physics.** NIRCam detects 181
       point sources while MIRI detects 3 — stars are bright in near-IR
       but the mid-IR is dominated by extended dust emission that suppresses
       point-source detection.
    """)

    st.markdown("---")
    st.subheader("Output files")
    files = [
        ("Images/sn1987a_rgb.png",             "NIRCam wide field RGB"),
        ("Images/sn1987a_comparison.png",       "NIRCam dynamic range comparison"),
        ("Images/sn1987a_enhanced_final.png",   "NIRCam FFT+CLAHE enhanced"),
        ("Images/alignment_result_nircam.png",         "ORB alignment result"),
        ("Images/fft_widefield_gentle.png",     "NIRCam FFT wide field"),
        ("Images/fft_sub320_banding.png",       "NIRCam sub320 banding removal"),
        ("Images/edge_detection_nircam.png",           "NIRCam edge detection"),
        ("Images/morphological_ops_nircam.png",        "NIRCam morphological ops"),
        ("Images/source_detection_nircam.png",         "NIRCam 181 sources"),
        ("Images/hough_circle_nircam.png",             "NIRCam Hough circle"),
        ("Images/pca_components_nircam.png",           "NIRCam PCA 6 components"),
        ("Images/miri_raw_enhanced.png",        "MIRI raw + enhanced RGB"),
        ("Images/miri_edge_detection.png",      "MIRI edge detection"),
        ("Images/miri_morphological_ops.png",   "MIRI morphological ops"),
        ("Images/miri_source_detection.png",    "MIRI 3 sources"),
        ("Images/miri_hough_circle.png",        "MIRI Hough circle"),
        ("Images/miri_pca_components.png",      "MIRI PCA components"),
        ("Images/miri_enhancement_pipeline.png",      "MIRI enhancement pipeline"),
        ("Images/miri_fft_power.png",                 "MIRI FFT power spectrum"),
        ("source_catalog_miri.csv",                  "MIRI source catalog"),
        ("enhancement_metrics_miri.json",            "PSNR + SSIM metrics"),
        ("source_catalog_nircam.csv",                  "NIRCam source catalog"),
        ("enhancement_metrics_nircam.json",            "PSNR + SSIM metrics"),
        ("pca_results_nircam.json",                    "PCA variance results"),
    ]
    for fname, desc in files:
        exists = Path(fname).exists()
        st.markdown(f"{'Created' if exists else 'Not Created'} `{fname}` — {desc}")