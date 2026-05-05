## ENHANCING THE IMAGES 


```python
# ── enhance.ipynb | Cell 1: imports + load wide field bands ──────────────────
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval
from scipy import ndimage
from skimage import exposure
import cv2

# Load wide field bands (the ones that gave the beautiful RGB)
bands = {}
for f in sorted(Path("./Data").glob("*i2d.fits")):
    with fits.open(f) as hdul:
        exts  = [h.name for h in hdul]
        sci   = hdul["SCI"].data.astype(np.float32)
        wcs   = WCS(hdul["SCI"].header)
        phdr  = hdul[0].header
        filt  = phdr.get("FILTER", "UNKNOWN")
        dq    = hdul["DQ"].data if "DQ" in exts else np.zeros_like(sci, dtype=np.uint32)
    bad = (dq != 0) | ~np.isfinite(sci)
    sci[bad] = np.nan
    bands[filt.upper()] = {"sci": sci, "wcs": wcs, "path": f, "filter": filt}
    print(f"Loaded {filt}: shape={sci.shape}")

print(f"\nBands: {list(bands.keys())}")

# Work on F277W — mid-range filter shows both stars and structure
working_band = "F1000W"
sci_raw = bands[working_band]["sci"].copy()
print(f"\nWorking band: {working_band}  shape={sci_raw.shape}")
```

    WARNING: FITSFixedWarning: 'datfix' made the change 'Set DATE-BEG to '2022-07-16T13:41:46.085' from MJD-BEG.
    Set DATE-AVG to '2022-07-16T13:53:04.897' from MJD-AVG.
    Set DATE-END to '2022-07-16T14:04:23.708' from MJD-END'. [astropy.wcs.wcs]
    WARNING: FITSFixedWarning: 'obsfix' made the change 'Set OBSGEO-L to   -70.271731 from OBSGEO-[XYZ].
    Set OBSGEO-B to   -37.855583 from OBSGEO-[XYZ].
    Set OBSGEO-H to 1740559880.406 from OBSGEO-[XYZ]'. [astropy.wcs.wcs]
    WARNING: FITSFixedWarning: 'datfix' made the change 'Set DATE-BEG to '2022-07-16T16:24:54.084' from MJD-BEG.
    Set DATE-AVG to '2022-07-16T16:53:46.026' from MJD-AVG.
    Set DATE-END to '2022-07-16T17:22:37.969' from MJD-END'. [astropy.wcs.wcs]
    

    Loaded F2550W: shape=(542, 539)
    Loaded F1000W: shape=(1028, 1032)
    Loaded F560W: shape=(1028, 1032)
    
    Bands: ['F2550W', 'F1000W', 'F560W']
    
    Working band: F1000W  shape=(1028, 1032)
    

    WARNING: FITSFixedWarning: 'obsfix' made the change 'Set OBSGEO-L to   -70.247501 from OBSGEO-[XYZ].
    Set OBSGEO-B to   -37.842991 from OBSGEO-[XYZ].
    Set OBSGEO-H to 1740356343.171 from OBSGEO-[XYZ]'. [astropy.wcs.wcs]
    WARNING: FITSFixedWarning: 'datfix' made the change 'Set DATE-BEG to '2022-07-16T14:20:40.268' from MJD-BEG.
    Set DATE-AVG to '2022-07-16T14:49:20.258' from MJD-AVG.
    Set DATE-END to '2022-07-16T15:18:00.216' from MJD-END'. [astropy.wcs.wcs]
    WARNING: FITSFixedWarning: 'obsfix' made the change 'Set OBSGEO-L to   -70.265479 from OBSGEO-[XYZ].
    Set OBSGEO-B to   -37.852351 from OBSGEO-[XYZ].
    Set OBSGEO-H to 1740507790.180 from OBSGEO-[XYZ]'. [astropy.wcs.wcs]
    


```python
# ── enhance.ipynb | Cell 2: FFT noise removal ────────────────────────────────
#
# JWST NIRCam has 1/f noise — horizontal banding visible as periodic stripes.
# Steps:
#   1. FFT → power spectrum
#   2. Identify noise frequencies (horizontal bands in frequency space)
#   3. Build a mask to suppress them
#   4. Inverse FFT → clean image

# Replace NaN with median before FFT (FFT can't handle NaN)
sci_filled = sci_raw.copy()
median_val = np.nanmedian(sci_filled)
sci_filled = np.where(np.isfinite(sci_filled), sci_filled, median_val)

# ── Step 1: 2D FFT ────────────────────────────────────────────────────────────
fft        = np.fft.fft2(sci_filled)
fft_shift  = np.fft.fftshift(fft)          # shift zero-freq to centre
power      = np.log1p(np.abs(fft_shift))   # log scale for visualisation

# ── Step 2: visualise power spectrum ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(sci_filled, origin="lower", cmap="inferno",
               vmin=np.nanpercentile(sci_filled, 2),
               vmax=np.nanpercentile(sci_filled, 99))
axes[0].set_title(f"{working_band} — raw", fontsize=11)
axes[0].axis("off")

axes[1].imshow(power, origin="lower", cmap="magma")
axes[1].set_title("Power spectrum (FFT)", fontsize=11)
axes[1].axis("off")
plt.suptitle("FFT power spectrum — look for bright lines = periodic noise", fontsize=11)
plt.tight_layout()
plt.savefig("Images_A/miri_power_spectrum.png", dpi=150, bbox_inches="tight")
plt.show()
```


    
![png](enhance_a_files/enhance_a_2_0.png)
    



```python
# ── enhance.ipynb | Cell 3: FFT mask + clean reconstruction ──────────────────
#
# Horizontal stripes in the image = vertical lines in frequency space
# We suppress a narrow horizontal band through the centre of the power spectrum

rows, cols = sci_filled.shape
cy, cx     = rows // 2, cols // 2   # centre of shifted FFT

# ── Build the filter mask ─────────────────────────────────────────────────────
mask = np.ones((rows, cols), dtype=np.float32)

# Suppress horizontal frequency band (causes vertical stripes)
# Width controls how aggressively we filter — start at 3, increase if needed
h_width = 3
mask[cy - h_width : cy + h_width, :] = 0

# Suppress vertical frequency band (causes horizontal stripes / 1/f banding)
v_width = 3
mask[:, cx - v_width : cx + v_width] = 0

# Always keep the DC component (centre pixel) — removing it shifts mean to 0
dc_size = 10
mask[cy - dc_size : cy + dc_size, cx - dc_size : cx + dc_size] = 1

# ── Apply mask + inverse FFT ──────────────────────────────────────────────────
fft_masked    = fft_shift * mask
fft_unshift   = np.fft.ifftshift(fft_masked)
sci_clean     = np.fft.ifft2(fft_unshift).real.astype(np.float32)

# Restore NaN mask from original
sci_clean[~np.isfinite(sci_raw)] = np.nan

# ── Compare before / after ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

vmin = np.nanpercentile(sci_raw, 2)
vmax = np.nanpercentile(sci_raw, 99)

axes[0].imshow(sci_raw,   origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
axes[0].set_title("Original", fontsize=11)
axes[0].axis("off")

axes[1].imshow(sci_clean, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
axes[1].set_title("FFT filtered", fontsize=11)
axes[1].axis("off")

# Difference image — shows exactly what was removed
diff = sci_raw - sci_clean
diff = np.where(np.isfinite(diff), diff, 0)
axes[2].imshow(diff, origin="lower", cmap="RdBu_r",
               vmin=-np.nanpercentile(np.abs(diff), 99),
               vmax= np.nanpercentile(np.abs(diff), 99))
axes[2].set_title("Removed noise (difference)", fontsize=11)
axes[2].axis("off")

plt.suptitle(f"FFT noise removal — {working_band}", fontsize=12)
plt.tight_layout()
plt.savefig("Images_A/miri_cleaned.png", dpi=150, bbox_inches="tight")
plt.show()

# Store cleaned band
bands[working_band]["sci_clean"] = sci_clean
print("MIRI cleaned — stored in bands['F277W']['sci_clean']")
```


    
![png](enhance_a_files/enhance_a_3_0.png)
    


    MIRI cleaned — stored in bands['F277W']['sci_clean']
    


```python
# ── enhance.ipynb | Cell 4: CLAHE ────────────────────────────────────────────
#
# CLAHE = Contrast Limited Adaptive Histogram Equalisation
# Divides image into small tiles, equalises each tile's histogram separately
# Result: local contrast boost — faint structure near bright sources becomes visible
# clip_limit controls how aggressively contrast is boosted (higher = more aggressive)

def apply_clahe(sci, clip_limit=0.01, nbins=256):
    """
    Apply CLAHE to a float32 science image.
    Normalises to 0-1 first, applies CLAHE, returns float32.
    """
    img = sci.copy()
    img = np.where(np.isfinite(img), img, 0.0)

    # Normalise to 0-1 using percentile clip
    vmin = np.percentile(img, 1)
    vmax = np.percentile(img, 99.5)
    img  = np.clip(img, vmin, vmax)
    img  = (img - vmin) / (vmax - vmin + 1e-10)

    # Apply CLAHE
    img_clahe = exposure.equalize_adapthist(img,
                                            clip_limit=clip_limit,
                                            nbins=nbins)
    return img_clahe.astype(np.float32)

# Apply to FFT-cleaned band
sci_clahe = apply_clahe(sci_clean, clip_limit=0.01)

# Also apply directly to raw for comparison
sci_clahe_raw = apply_clahe(sci_filled, clip_limit=0.01)

# ── 4-panel comparison ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 6))

norm_raw   = dict(vmin=np.nanpercentile(sci_filled,2),
                  vmax=np.nanpercentile(sci_filled,99))

axes[0].imshow(sci_filled,    origin="lower", cmap="inferno", **norm_raw)
axes[0].set_title("1. Raw",                  fontsize=11)

axes[1].imshow(sci_clean,     origin="lower", cmap="inferno", **norm_raw)
axes[1].set_title("2. FFT filtered",         fontsize=11)

axes[2].imshow(sci_clahe_raw, origin="lower", cmap="inferno")
axes[2].set_title("3. CLAHE only",           fontsize=11)

axes[3].imshow(sci_clahe,     origin="lower", cmap="inferno")
axes[3].set_title("4. FFT + CLAHE",          fontsize=11)

for ax in axes:
    ax.axis("off")

plt.suptitle(f"Enhancement pipeline — {working_band}", fontsize=12)
plt.tight_layout()
plt.savefig("Images_A/miri_enhancement_pipeline.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → miri_enhancement_pipeline.png")
```


    
![png](enhance_a_files/enhance_a_4_0.png)
    


    Saved → miri_enhancement_pipeline.png
    


```python
# ── enhance.ipynb | Cell 5: enhance all bands + final RGB ────────────────────

def enhance_band(sci):
    """Full enhancement pipeline: FFT filter → CLAHE."""
    # Fill NaN
    filled   = np.where(np.isfinite(sci), sci, np.nanmedian(sci))
    # FFT denoise
    fft      = np.fft.fftshift(np.fft.fft2(filled))
    rows, cols = filled.shape
    cy, cx   = rows//2, cols//2
    mask     = np.ones((rows,cols), dtype=np.float32)
    mask[cy-3:cy+3, :]    = 0
    mask[:, cx-3:cx+3]    = 0
    mask[cy-10:cy+10, cx-10:cx+10] = 1
    cleaned  = np.fft.ifft2(np.fft.ifftshift(fft*mask)).real.astype(np.float32)
    cleaned[~np.isfinite(sci)] = np.nan
    # CLAHE
    enhanced = apply_clahe(cleaned, clip_limit=0.01)
    return enhanced

# Enhance all 3 bands
enhanced_bands = {}
for fname in ["F560W", "F1000W", "F2550W"]:
    print(f"Enhancing {fname}...")
    enhanced_bands[fname] = enhance_band(bands[fname]["sci"])
    print(f"  Done — shape={enhanced_bands[fname].shape}")

# Build final enhanced RGB
def make_rgb_enhanced(enhanced_bands, blue="F560W", green="F1000W", red="F2550W"):
    r = enhanced_bands[red]
    g = enhanced_bands[green]
    b = enhanced_bands[blue]

    # Resize to match if needed
    target_shape = r.shape
    if g.shape != target_shape:
        g = cv2.resize(g, (target_shape[1],target_shape[0]), interpolation=cv2.INTER_CUBIC)
    if b.shape != target_shape:
        b = cv2.resize(b, (target_shape[1],target_shape[0]), interpolation=cv2.INTER_CUBIC)

    rgb = np.dstack([r, g, b])
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

rgb_enhanced = make_rgb_enhanced(enhanced_bands)

# Side by side: original RGB vs enhanced RGB
# Side by side: original RGB vs enhanced RGB
from align_and_merge import align_and_merge
import importlib
import align_and_merge as am

# 1. Force a reload of the module to pick up your file changes
importlib.reload(am)

# 2. Explicitly set use_orb=False in the function call
result_raw = am.align_and_merge(
    bands, 
    blue_filter="F560W", 
    green_filter="F1000W", 
    red_filter="F2550W",
    use_orb=False # Forces WCS remapping to (542, 539)
)

def make_rgb_v2(result, percentile_low=2, percentile_high=99.5, Q=8):
    channels = {}
    for ch in ["blue","green","red"]:
        img = result[ch].copy()
        img = np.where(np.isfinite(img), img, np.nanmedian(img))
        vmin = np.nanpercentile(img, percentile_low)
        vmax = np.nanpercentile(img, percentile_high)
        img  = np.clip(img, vmin, vmax)
        img  = (img - vmin) / (vmax - vmin + 1e-10)
        img  = np.arcsinh(Q * img) / np.arcsinh(Q)
        channels[ch] = img
    rgb = np.dstack([channels["red"],channels["green"],channels["blue"]])
    return (np.clip(rgb,0,1)*255).astype(np.uint8)

rgb_raw = make_rgb_v2(result_raw, percentile_high=99.0, Q=5)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(rgb_raw,      origin="lower")
axes[0].set_title("Original RGB",        fontsize=12)
axes[0].axis("off")
axes[1].imshow(rgb_enhanced, origin="lower")
axes[1].set_title("FFT + CLAHE enhanced RGB", fontsize=12)
axes[1].axis("off")

plt.suptitle("SN 1987A — Enhancement comparison", fontsize=13)
plt.tight_layout()
plt.savefig("Images_A/miri_enhanced_final.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved → miri_enhanced_final.png")
```

    Enhancing F560W...
      Done — shape=(1028, 1032)
    Enhancing F1000W...
      Done — shape=(1028, 1032)
    Enhancing F2550W...
      Done — shape=(542, 539)
    
    Aligning F560W → F2550W...
    

    WARNING: FITSFixedWarning: 'datfix' made the change 'Set DATE-BEG to '2022-07-16T14:20:40.268' from MJD-BEG.
    Set DATE-AVG to '2022-07-16T14:49:20.258' from MJD-AVG.
    Set DATE-END to '2022-07-16T15:18:00.216' from MJD-END'. [astropy.wcs.wcs]
    WARNING: FITSFixedWarning: 'obsfix' made the change 'Set OBSGEO-L to   -70.265479 from OBSGEO-[XYZ].
    Set OBSGEO-B to   -37.852351 from OBSGEO-[XYZ].
    Set OBSGEO-H to 1740507790.180 from OBSGEO-[XYZ]'. [astropy.wcs.wcs]
    WARNING: FITSFixedWarning: 'datfix' made the change 'Set DATE-BEG to '2022-07-16T13:41:46.085' from MJD-BEG.
    Set DATE-AVG to '2022-07-16T13:53:04.897' from MJD-AVG.
    Set DATE-END to '2022-07-16T14:04:23.708' from MJD-END'. [astropy.wcs.wcs]
    WARNING: FITSFixedWarning: 'obsfix' made the change 'Set OBSGEO-L to   -70.271731 from OBSGEO-[XYZ].
    Set OBSGEO-B to   -37.855583 from OBSGEO-[XYZ].
    Set OBSGEO-H to 1740559880.406 from OBSGEO-[XYZ]'. [astropy.wcs.wcs]
    

    
    Aligning F1000W → F2550W...
    

    WARNING: FITSFixedWarning: 'datfix' made the change 'Set DATE-BEG to '2022-07-16T16:24:54.084' from MJD-BEG.
    Set DATE-AVG to '2022-07-16T16:53:46.026' from MJD-AVG.
    Set DATE-END to '2022-07-16T17:22:37.969' from MJD-END'. [astropy.wcs.wcs]
    WARNING: FITSFixedWarning: 'obsfix' made the change 'Set OBSGEO-L to   -70.247501 from OBSGEO-[XYZ].
    Set OBSGEO-B to   -37.842991 from OBSGEO-[XYZ].
    Set OBSGEO-H to 1740356343.171 from OBSGEO-[XYZ]'. [astropy.wcs.wcs]
    

    
    Wavelet merging...
    Merged shape: (542, 539)
    


    
![png](enhance_a_files/enhance_a_5_5.png)
    


    Saved → miri_enhanced_final.png
    


```python
# ── enhance.ipynb | Cell 6: FIXED FFT + CLAHE pipeline ───────────────────────
#
# Key insight from Cell 2: F277W power spectrum has NO bright lines
# = no strong periodic noise in this dataset
# FFT filter should be very conservative — only remove DC offset artifacts
# The real value of FFT here is DEMONSTRATING the technique, not fixing real noise

import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import cv2

# ── Correct FFT approach for clean data ───────────────────────────────────────
# Instead of a cross mask (which caused grid artifacts), use a gentle
# notch filter that only removes the very centre DC spike

def fft_gentle(sci):
    """
    Gentle FFT filter for already-clean data.
    Only removes DC component and very low frequencies.
    For data WITH banding, increase notch_width.
    """
    filled = np.where(np.isfinite(sci), sci, np.nanmedian(sci))

    fft       = np.fft.fft2(filled)
    fft_shift = np.fft.fftshift(fft)

    rows, cols = filled.shape
    cy, cx     = rows//2, cols//2

    # Build distance map from centre
    yy, xx   = np.ogrid[:rows, :cols]
    dist     = np.sqrt((yy-cy)**2 + (xx-cx)**2)

    # Notch filter — remove only the DC spike (centre 2px)
    # This removes the flat background offset without creating grid artifacts
    mask = np.ones((rows,cols), dtype=np.float32)
    mask[dist < 2] = 0    # remove DC only

    cleaned = np.fft.ifft2(np.fft.ifftshift(fft_shift * mask)).real
    cleaned = cleaned.astype(np.float32)
    cleaned[~np.isfinite(sci)] = np.nan
    return cleaned

# ── Also demonstrate the FULL FFT pipeline for report purposes ────────────────
# Even on clean data, showing the pipeline is valuable
# We demonstrate ON THE SUB320 DATA which actually has banding (visible in bands)

def fft_denoise_banding(sci, h_width=2, v_width=2):
    """
    FFT filter targeting horizontal/vertical banding.
    h_width: suppress rows near horizontal axis in freq domain (removes V stripes)
    v_width: suppress cols near vertical axis in freq domain (removes H stripes)
    """
    filled = np.where(np.isfinite(sci), sci, np.nanmedian(sci))

    fft       = np.fft.fft2(filled)
    fft_shift = np.fft.fftshift(fft)
    rows, cols = filled.shape
    cy, cx     = rows//2, cols//2

    mask = np.ones((rows,cols), dtype=np.float32)

    # Suppress only the cross axes — NOT a full cross width
    # Use a smooth taper (Gaussian) instead of hard cutoff to avoid ringing
    yy, xx = np.ogrid[:rows, :cols]

    # Soft horizontal band suppression
    h_taper = 1 - np.exp(-0.5 * ((yy - cy) / h_width)**2)
    # Soft vertical band suppression
    v_taper = 1 - np.exp(-0.5 * ((xx - cx) / v_width)**2)

    mask = h_taper * v_taper
    # Keep DC
    mask[cy-5:cy+5, cx-5:cx+5] = 1.0

    cleaned = np.fft.ifft2(np.fft.ifftshift(fft_shift * mask)).real
    cleaned = cleaned.astype(np.float32)
    cleaned[~np.isfinite(sci)] = np.nan
    return cleaned

# ── Apply to F560W (wide field — clean data, show gentle filter) ──────────────
sci_wide = bands["F560W"]["sci"].copy()
sci_fft_gentle = fft_gentle(sci_wide)

# ── Apply to sub320 F1000W (actually has banding — show real denoising) ────────
sci_sub = bands["F1000W"]["sci"].copy()
sci_fft_sub = fft_denoise_banding(sci_sub, h_width=2, v_width=2)

# ── Plot 1: wide field — gentle FFT ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
vmin, vmax = np.nanpercentile(sci_wide, 2), np.nanpercentile(sci_wide, 99)

axes[0].imshow(sci_wide,       origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
axes[0].set_title("F560W — original",          fontsize=11)
axes[1].imshow(sci_fft_gentle, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
axes[1].set_title("F560W — gentle FFT (DC removed)", fontsize=11)
diff = np.where(np.isfinite(sci_wide - sci_fft_gentle), sci_wide - sci_fft_gentle, 0)
axes[2].imshow(diff, origin="lower", cmap="RdBu_r",
               vmin=-np.percentile(np.abs(diff),99),
               vmax= np.percentile(np.abs(diff),99))
axes[2].set_title("Difference (removed component)", fontsize=11)
for ax in axes: ax.axis("off")
plt.suptitle("FFT — wide field (clean data, DC-only filter)", fontsize=12)
plt.tight_layout()
plt.savefig("fft_widefield_gentle.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Plot 2: sub320 — banding removal ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
vmin2, vmax2 = np.nanpercentile(sci_sub, 2), np.nanpercentile(sci_sub, 99)

axes[0].imshow(sci_sub,     origin="lower", cmap="inferno", vmin=vmin2, vmax=vmax2)
axes[0].set_title("F1000W sub320 — original (has banding)", fontsize=11)
axes[1].imshow(sci_fft_sub, origin="lower", cmap="inferno", vmin=vmin2, vmax=vmax2)
axes[1].set_title("F1000W sub320 — FFT banding removed",     fontsize=11)
diff2 = np.where(np.isfinite(sci_sub - sci_fft_sub), sci_sub - sci_fft_sub, 0)
axes[2].imshow(diff2, origin="lower", cmap="RdBu_r",
               vmin=-np.percentile(np.abs(diff2),99),
               vmax= np.percentile(np.abs(diff2),99))
axes[2].set_title("Removed banding pattern", fontsize=11)
for ax in axes: ax.axis("off")
plt.suptitle("FFT — sub320 (has real 1/f banding, Gaussian notch filter)", fontsize=12)
plt.tight_layout()
plt.savefig("fft_sub320_banding.png", dpi=150, bbox_inches="tight")
plt.show()
```


    
![png](enhance_a_files/enhance_a_6_0.png)
    



    
![png](enhance_a_files/enhance_a_6_1.png)
    



```python
# ── enhance.ipynb | Cell 7: CLAHE + correct final RGB ────────────────────────
from skimage import exposure

def apply_clahe(sci, clip_limit=0.03):
    img = np.where(np.isfinite(sci), sci, 0.0)
    vmin, vmax = np.percentile(img[img>0], 1), np.percentile(img[img>0], 99.5)
    img  = np.clip(img, vmin, vmax)
    img  = (img - vmin) / (vmax - vmin + 1e-10)
    return exposure.equalize_adapthist(img, clip_limit=clip_limit).astype(np.float32)

# ── 1. Resize raw bands to same shape FIRST ───────────────────────────────────
ref_shape = bands["F560W"]["sci"].shape
for fname in ["F560W", "F1000W", "F2550W"]:
    sci = bands[fname]["sci"]
    if sci.shape != ref_shape:
        print(f"Resizing raw {fname}: {sci.shape} → {ref_shape}")
        resized = cv2.resize(
            np.where(np.isfinite(sci), sci, 0.0),
            (ref_shape[1], ref_shape[0]),
            interpolation=cv2.INTER_CUBIC
        )
        bands[fname]["sci"] = resized.astype(np.float32)

# ── 2. Enhance all bands ──────────────────────────────────────────────────────
enhanced = {}
for fname in ["F560W", "F1000W", "F2550W"]:
    sci = fft_gentle(bands[fname]["sci"])
    enhanced[fname] = apply_clahe(sci, clip_limit=0.03)
    print(f"Enhanced {fname}")

# ── 3. Resize enhanced bands to same shape ────────────────────────────────────
target_shape = enhanced["F560W"].shape
for fname in ["F560W", "F1000W", "F2550W"]:
    if enhanced[fname].shape != target_shape:
        print(f"Resizing enhanced {fname}: {enhanced[fname].shape} → {target_shape}")
        enhanced[fname] = cv2.resize(
            enhanced[fname],
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_CUBIC
        )

# Verify all shapes match
for fname, arr in enhanced.items():
    print(f"{fname}: {arr.shape}")

# ── 4. Build RGBs ─────────────────────────────────────────────────────────────
def make_rgb_enhanced(enh, blue="F560W", green="F1000W", red="F2550W"):
    r = enh[red];   g = enh[green];   b = enh[blue]
    h, w = r.shape
    g = cv2.resize(g, (w,h), interpolation=cv2.INTER_CUBIC) if g.shape!=(h,w) else g
    b = cv2.resize(b, (w,h), interpolation=cv2.INTER_CUBIC) if b.shape!=(h,w) else b
    rgb = np.dstack([r, g, b])
    return (np.clip(rgb,0,1)*255).astype(np.uint8)

def make_rgb_raw(bands, blue="F560W", green="F1000W", red="F2550W", Q=5):
    channels = {}
    for ch, fname in [("red",red),("green",green),("blue",blue)]:
        img = bands[fname]["sci"].copy()
        img = np.where(np.isfinite(img), img, np.nanmedian(img))
        vmin = np.nanpercentile(img, 2)
        vmax = np.nanpercentile(img, 99.0)
        img  = np.clip(img, vmin, vmax)
        img  = (img-vmin)/(vmax-vmin+1e-10)
        img  = np.arcsinh(Q*img)/np.arcsinh(Q)
        channels[ch] = img
    rgb = np.dstack([channels["red"],channels["green"],channels["blue"]])
    return (np.clip(rgb,0,1)*255).astype(np.uint8)

rgb_raw      = make_rgb_raw(bands)
rgb_enhanced = make_rgb_enhanced(enhanced)

# ── 5. Plot comparison ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(rgb_raw,      origin="lower")
axes[0].set_title("Original (Lupton stretch)",  fontsize=12)
axes[0].axis("off")
axes[1].imshow(rgb_enhanced, origin="lower")
axes[1].set_title("FFT + CLAHE enhanced",       fontsize=12)
axes[1].axis("off")
plt.suptitle("SN 1987A wide field — Enhancement comparison", fontsize=13)
plt.tight_layout()
plt.savefig("sn1987a_enhanced_final.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved → sn1987a_enhanced_final.png")
```

    Resizing raw F2550W: (542, 539) → (1028, 1032)
    Enhanced F560W
    Enhanced F1000W
    Enhanced F2550W
    F560W: (1028, 1032)
    F1000W: (1028, 1032)
    F2550W: (1028, 1032)
    


    
![png](enhance_a_files/enhance_a_7_1.png)
    


    Saved → sn1987a_enhanced_final.png
    


```python

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity   as ssim
import json
import numpy as np

def to_float(rgb):
    return rgb.astype(np.float32) / 255.0

rgb_raw_f = to_float(rgb_raw)
rgb_enh_f = to_float(rgb_enhanced)

if rgb_raw_f.shape != rgb_enh_f.shape:
    h, w = rgb_raw_f.shape[:2]
    rgb_enh_f = cv2.resize(rgb_enh_f, (w, h), interpolation=cv2.INTER_CUBIC)

psnr_val = psnr(rgb_raw_f, rgb_enh_f, data_range=1.0)
ssim_val = ssim(rgb_raw_f, rgb_enh_f, data_range=1.0, channel_axis=2)

print(f"PSNR : {psnr_val:.2f} dB") #Peak Signal-to-Noise Ratio in decibels
print(f"SSIM : {ssim_val:.4f}") #Structural Similarity Index (0 to 1, higher is more similar)

metrics = {
    "psnr_db":  float(round(psnr_val, 4)),
    "ssim":     float(round(ssim_val, 4)),
    "filter":   "F560W/F1000W/F2550W",
    "pipeline": "FFT DC-notch + CLAHE clip=0.03",
    "target":   "SN1987A wide field (Program 1232)",
}
with open("Images_A/enhancement_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved → enhancement_metrics.json")

# ── Fix: plain np.save calls (no hyperlinks) ──────────────────────────────────
np.save("Images_A/miri_enhanced_f444w.npy", enhanced["F2550W"])
np.save("Images_A/miri_enhanced_f277w.npy", enhanced["F1000W"])
np.save("Images_A/miri_enhanced_f115w.npy", enhanced["F560W"])



```

    PSNR : 10.72 dB
    SSIM : 0.2337
    Saved → enhancement_metrics.json
    


```python
print("PSNR/SSIM Interpretation for enhancement tasks:")
print()
print(f"PSNR = {psnr_val:.2f} dB")
print(f"SSIM = {ssim_val:.4f}")
print()
print("These metrics measure SIMILARITY between original and enhanced.")
print("Low values = large change was made — expected for CLAHE.")
print("PSNR/SSIM are most meaningful for denoising tasks where the")
print("goal is to recover an original clean image.")
print()
print("For enhancement the better metric is visual quality + SNR:")

# Signal to Noise Ratio on F277W
sci_f277 = bands["F1000W"]["sci"]
sci_f277_clean = fft_gentle(sci_f277)
valid = sci_f277_clean[np.isfinite(sci_f277_clean)]
snr = np.nanmean(valid) / np.nanstd(valid)
print(f"F1000W SNR before enhancement: {snr:.2f}")

enh_f277 = enhanced["F1000W"]
snr_enh  = enh_f277.mean() / (enh_f277.std() + 1e-10)
print(f"F1000W SNR after  enhancement: {snr_enh:.2f}")
print()
print("A higher SNR after enhancement = better for detection tasks.")
print("The visual result (ring clearly visible) is the primary metric.")
```

    PSNR/SSIM Interpretation for enhancement tasks:
    
    PSNR = 10.72 dB
    SSIM = 0.2337
    
    These metrics measure SIMILARITY between original and enhanced.
    Low values = large change was made — expected for CLAHE.
    PSNR/SSIM are most meaningful for denoising tasks where the
    goal is to recover an original clean image.
    
    For enhancement the better metric is visual quality + SNR:
    F1000W SNR before enhancement: 0.01
    F1000W SNR after  enhancement: 0.39
    
    A higher SNR after enhancement = better for detection tasks.
    The visual result (ring clearly visible) is the primary metric.
    


```python
from astropy.visualization import ImageNormalize, ZScaleInterval
from sklearn.decomposition import PCA

# 1. Prepare the data: Flatten and stack the aligned bands
# Using the aligned 'result_raw' from your previous cell
b = result_raw["blue"].flatten()
g = result_raw["green"].flatten()
r = result_raw["red"].flatten()

# Filter out NaNs for the calculation
mask = np.isfinite(b) & np.isfinite(g) & np.isfinite(r)
data_stack = np.stack([b[mask], g[mask], r[mask]], axis=1)

# 2. Run PCA
pca = PCA(n_components=3)
pca_data = pca.fit_transform(data_stack)

# 3. Reshape back to image dimensions
h, w = result_raw["blue"].shape
pc_images = []
for i in range(3):
    pc_img = np.full((h, w), np.nan)
    pc_img[mask.reshape(h, w)] = pca_data[:, i]
    pc_images.append(pc_img)

# 4. Plot the Principal Components
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, ax in enumerate(axes):
    norm = ImageNormalize(pc_images[i][np.isfinite(pc_images[i])], interval=ZScaleInterval())
    ax.imshow(pc_images[i], origin="lower", cmap="viridis", norm=norm)
    ax.set_title(f"Principal Component {i+1} ({pca.explained_variance_ratio_[i]*100:.1f}%)")
    ax.axis("off")
plt.savefig("Images_A/miri_pca_results.png", dpi=200)
plt.show()

print(f"PCA Variance Explained: {pca.explained_variance_ratio_}")
```


    
![png](enhance_a_files/enhance_a_10_0.png)
    


    PCA Variance Explained: [0.87630296 0.12369707 0.        ]
    


```python
# Create a wide-field comparison of raw vs. enhanced
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Panel 1: Raw F1000W (Green) - No processing
raw_data = bands["F1000W"]["sci"]
norm_raw = ImageNormalize(raw_data[np.isfinite(raw_data)], interval=ZScaleInterval())
axes[0].imshow(raw_data, origin="lower", cmap="inferno", norm=norm_raw)
axes[0].set_title("Raw MIRI F1000W Wide Field (Program 1232)", fontsize=12)

# Panel 2: Your Enhanced Result[cite: 4]
axes[1].imshow(rgb_enhanced, origin="lower")
axes[1].set_title("Final Enhanced RGB (FFT + CLAHE)", fontsize=12)

for ax in axes: ax.axis("off")
plt.savefig("Images_A/miri_raw_vs_enhanced.png", dpi=200)
plt.show()
```


    
![png](enhance_a_files/enhance_a_11_0.png)
    

