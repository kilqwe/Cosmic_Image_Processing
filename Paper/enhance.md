## ENHANCING THE IMAGES 


```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval
from scipy import ndimage
from skimage import exposure
import cv2

bands = {}
for f in sorted(Path("./jwst_data/sn1987a").glob("*i2d.fits")):
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
working_band = "F277W"
sci_raw = bands[working_band]["sci"].copy()
print(f"\nWorking band: {working_band}  shape={sci_raw.shape}")
```

    WARNING: FITSFixedWarning: 'datfix' made the change 'Set DATE-BEG to '2022-07-16T12:51:34.481' from MJD-BEG.
    Set DATE-AVG to '2022-07-16T13:01:25.016' from MJD-AVG.
    Set DATE-END to '2022-07-16T13:11:15.551' from MJD-END'. [astropy.wcs.wcs]
    WARNING: FITSFixedWarning: 'obsfix' made the change 'Set OBSGEO-L to   -70.279077 from OBSGEO-[XYZ].
    Set OBSGEO-B to   -37.859367 from OBSGEO-[XYZ].
    Set OBSGEO-H to 1740620703.584 from OBSGEO-[XYZ]'. [astropy.wcs.wcs]
    WARNING: FITSFixedWarning: 'datfix' made the change 'Set DATE-BEG to '2022-07-16T13:41:51.566' from MJD-BEG.
    Set DATE-AVG to '2022-07-16T13:52:43.829' from MJD-AVG.
    Set DATE-END to '2022-07-16T14:03:30.716' from MJD-END'. [astropy.wcs.wcs]
    WARNING: FITSFixedWarning: 'obsfix' made the change 'Set OBSGEO-L to   -70.271792 from OBSGEO-[XYZ].
    Set OBSGEO-B to   -37.855615 from OBSGEO-[XYZ].
    Set OBSGEO-H to 1740560385.068 from OBSGEO-[XYZ]'. [astropy.wcs.wcs]
    

    Loaded F115W: shape=(4370, 4389)
    Loaded F277W: shape=(2122, 2120)
    

    WARNING: FITSFixedWarning: 'datfix' made the change 'Set DATE-BEG to '2022-07-16T13:16:37.648' from MJD-BEG.
    Set DATE-AVG to '2022-07-16T13:26:25.495' from MJD-AVG.
    Set DATE-END to '2022-07-16T13:36:07.966' from MJD-END'. [astropy.wcs.wcs]
    WARNING: FITSFixedWarning: 'obsfix' made the change 'Set OBSGEO-L to   -70.275447 from OBSGEO-[XYZ].
    Set OBSGEO-B to   -37.857499 from OBSGEO-[XYZ].
    Set OBSGEO-H to 1740590699.368 from OBSGEO-[XYZ]'. [astropy.wcs.wcs]
    WARNING: FITSFixedWarning: 'datfix' made the change 'Set DATE-BEG to '2022-09-02T13:18:12.622' from MJD-BEG.
    Set DATE-AVG to '2022-09-02T13:35:36.019' from MJD-AVG.
    Set DATE-END to '2022-09-02T13:52:59.331' from MJD-END'. [astropy.wcs.wcs]
    WARNING: FITSFixedWarning: 'obsfix' made the change 'Set OBSGEO-L to   -46.694797 from OBSGEO-[XYZ].
    Set OBSGEO-B to   -16.371046 from OBSGEO-[XYZ].
    Set OBSGEO-H to 1478832142.256 from OBSGEO-[XYZ]'. [astropy.wcs.wcs]
    WARNING: FITSFixedWarning: 'datfix' made the change 'Set DATE-BEG to '2022-09-02T13:57:07.348' from MJD-BEG.
    Set DATE-AVG to '2022-09-02T14:12:58.890' from MJD-AVG.
    Set DATE-END to '2022-09-02T14:28:49.345' from MJD-END'. [astropy.wcs.wcs]
    WARNING: FITSFixedWarning: 'obsfix' made the change 'Set OBSGEO-L to   -46.668561 from OBSGEO-[XYZ].
    Set OBSGEO-B to   -16.348296 from OBSGEO-[XYZ].
    Set OBSGEO-H to 1478590370.632 from OBSGEO-[XYZ]'. [astropy.wcs.wcs]
    

    Loaded F444W: shape=(2108, 2119)
    Loaded F150W: shape=(795, 810)
    Loaded F200W: shape=(794, 809)
    Loaded F444W: shape=(326, 325)
    
    Bands: ['F115W', 'F277W', 'F444W', 'F150W', 'F200W']
    
    Working band: F277W  shape=(2122, 2120)
    

    WARNING: FITSFixedWarning: 'datfix' made the change 'Set DATE-BEG to '2022-09-02T13:57:06.836' from MJD-BEG.
    Set DATE-AVG to '2022-09-02T14:12:58.844' from MJD-AVG.
    Set DATE-END to '2022-09-02T14:28:48.449' from MJD-END'. [astropy.wcs.wcs]
    WARNING: FITSFixedWarning: 'obsfix' made the change 'Set OBSGEO-L to   -46.668572 from OBSGEO-[XYZ].
    Set OBSGEO-B to   -16.348305 from OBSGEO-[XYZ].
    Set OBSGEO-H to 1478590470.575 from OBSGEO-[XYZ]'. [astropy.wcs.wcs]
    


```python
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

# 2D FFT 
fft        = np.fft.fft2(sci_filled)
fft_shift  = np.fft.fftshift(fft)          # shift zero-freq to centre
power      = np.log1p(np.abs(fft_shift))   # log scale for visualisation

# visualise power spectrum
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
plt.savefig("fft_power_spectrum.png", dpi=150, bbox_inches="tight")
plt.show()
```


    
![png](enhance_files/enhance_2_0.png)
    



```python

# Horizontal stripes in the image = vertical lines in frequency space
# We suppress a narrow horizontal band through the centre of the power spectrum

rows, cols = sci_filled.shape
cy, cx     = rows // 2, cols // 2   # centre of shifted FFT

#Build the filter mask
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

# Apply mask + inverse FFT 
fft_masked    = fft_shift * mask
fft_unshift   = np.fft.ifftshift(fft_masked)
sci_clean     = np.fft.ifft2(fft_unshift).real.astype(np.float32)

# Restore NaN mask from original
sci_clean[~np.isfinite(sci_raw)] = np.nan

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
plt.savefig("fft_cleaned.png", dpi=150, bbox_inches="tight")
plt.show()

# Store cleaned band
bands[working_band]["sci_clean"] = sci_clean
print("FFT cleaned — stored in bands['F277W']['sci_clean']")
```


    
![png](enhance_files/enhance_3_0.png)
    


    FFT cleaned — stored in bands['F277W']['sci_clean']
    


```python
# CLAHE = Contrast Limited Adaptive Histogram Equalisation Normalises to 0-1 first, applies CLAHE, returns float32.
# Divides image into small tiles, equalises each tile's histogram separately
# Result: local contrast boost — faint structure near bright sources becomes visible
# clip_limit controls how aggressively contrast is boosted (higher = more aggressive)

def apply_clahe(sci, clip_limit=0.01, nbins=256):
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

# 4-panel comparison
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
plt.savefig("enhancement_pipeline.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → enhancement_pipeline.png")
```


    
![png](enhance_files/enhance_4_0.png)
    


    Saved → enhancement_pipeline.png
    


```python
def enhance_band(sci):
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
for fname in ["F115W", "F277W", "F444W"]:
    print(f"Enhancing {fname}...")
    enhanced_bands[fname] = enhance_band(bands[fname]["sci"])
    print(f"  Done — shape={enhanced_bands[fname].shape}")

# Build final enhanced RGB
def make_rgb_enhanced(enhanced_bands, blue="F115W", green="F277W", red="F444W"):
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
from align_and_merge import align_and_merge
result_raw = align_and_merge(bands, blue_filter="F115W",
                             green_filter="F277W", red_filter="F444W")

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
plt.savefig("sn1987a_enhanced_final.png", dpi=200, bbox_inches="tight")
plt.show()
print("Saved → sn1987a_enhanced_final.png")
```

    Enhancing F115W...
      Done — shape=(4370, 4389)
    Enhancing F277W...
      Done — shape=(2122, 2120)
    Enhancing F444W...
      Done — shape=(326, 325)
    
    Aligning F115W → F444W...
    

    c:\Users\shrey\Desktop\CV_JWST_imagery\align_and_merge.py:19: RuntimeWarning: invalid value encountered in cast
      return (scaled * 255).astype(np.uint8)
    

    Keypoints — reference: 3244  moving: 5000
     Matches: 787 total → 118 kept
    Homography inliers: 5 / 118
      ORB OK
    
    Aligning F277W → F444W...
    Keypoints — reference: 3244  moving: 5000
     Matches: 952 total → 142 kept
    Homography inliers: 9 / 142
      ORB OK
    
    Wavelet merging...
    Merged shape: (326, 325)
    


    
![png](enhance_files/enhance_5_3.png)
    


    Saved → sn1987a_enhanced_final.png
    


```python
# F277W power spectrum has NO bright lines = no strong periodic noise in this dataset
# FFT filter should be very conservative — only remove DC offset artifacts
# The real value of FFT here is DEMONSTRATING the technique, not fixing real noise
# Gentle FFT filter for already-clean data.Only removes DC component and very low frequencies.For data WITH banding, increase notch_width.

import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import cv2


def fft_gentle(sci):

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


def fft_denoise_banding(sci, h_width=2, v_width=2):
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

# Apply to F277W (wide field — clean data, show gentle filter) 
sci_wide = bands["F277W"]["sci"].copy()
sci_fft_gentle = fft_gentle(sci_wide)

# Apply to sub320 F150W (actually has banding — show real denoising)
sci_sub = bands["F150W"]["sci"].copy()
sci_fft_sub = fft_denoise_banding(sci_sub, h_width=2, v_width=2)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
vmin, vmax = np.nanpercentile(sci_wide, 2), np.nanpercentile(sci_wide, 99)

axes[0].imshow(sci_wide,       origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
axes[0].set_title("F277W — original",          fontsize=11)
axes[1].imshow(sci_fft_gentle, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
axes[1].set_title("F277W — gentle FFT (DC removed)", fontsize=11)
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

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
vmin2, vmax2 = np.nanpercentile(sci_sub, 2), np.nanpercentile(sci_sub, 99)

axes[0].imshow(sci_sub,     origin="lower", cmap="inferno", vmin=vmin2, vmax=vmax2)
axes[0].set_title("F150W sub320 — original (has banding)", fontsize=11)
axes[1].imshow(sci_fft_sub, origin="lower", cmap="inferno", vmin=vmin2, vmax=vmax2)
axes[1].set_title("F150W sub320 — FFT banding removed",     fontsize=11)
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


    
![png](enhance_files/enhance_6_0.png)
    



    
![png](enhance_files/enhance_6_1.png)
    



```python
from skimage import exposure

def apply_clahe(sci, clip_limit=0.03):
    img = np.where(np.isfinite(sci), sci, 0.0)
    vmin, vmax = np.percentile(img[img>0], 1), np.percentile(img[img>0], 99.5)
    img  = np.clip(img, vmin, vmax)
    img  = (img - vmin) / (vmax - vmin + 1e-10)
    return exposure.equalize_adapthist(img, clip_limit=clip_limit).astype(np.float32)


ref_shape = bands["F277W"]["sci"].shape
for fname in ["F115W", "F277W", "F444W"]:
    sci = bands[fname]["sci"]
    if sci.shape != ref_shape:
        print(f"Resizing raw {fname}: {sci.shape} → {ref_shape}")
        resized = cv2.resize(
            np.where(np.isfinite(sci), sci, 0.0),
            (ref_shape[1], ref_shape[0]),
            interpolation=cv2.INTER_CUBIC
        )
        bands[fname]["sci"] = resized.astype(np.float32)


enhanced = {}
for fname in ["F115W", "F277W", "F444W"]:
    sci = fft_gentle(bands[fname]["sci"])
    enhanced[fname] = apply_clahe(sci, clip_limit=0.03)
    print(f"Enhanced {fname}")

target_shape = enhanced["F277W"].shape
for fname in ["F115W", "F277W", "F444W"]:
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

def make_rgb_enhanced(enh, blue="F115W", green="F277W", red="F444W"):
    r = enh[red];   g = enh[green];   b = enh[blue]
    h, w = r.shape
    g = cv2.resize(g, (w,h), interpolation=cv2.INTER_CUBIC) if g.shape!=(h,w) else g
    b = cv2.resize(b, (w,h), interpolation=cv2.INTER_CUBIC) if b.shape!=(h,w) else b
    rgb = np.dstack([r, g, b])
    return (np.clip(rgb,0,1)*255).astype(np.uint8)

def make_rgb_raw(bands, blue="F115W", green="F277W", red="F444W", Q=5):
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

    Resizing raw F115W: (4370, 4389) → (2122, 2120)
    Resizing raw F444W: (326, 325) → (2122, 2120)
    Enhanced F115W
    Enhanced F277W
    Enhanced F444W
    F115W: (2122, 2120)
    F277W: (2122, 2120)
    F444W: (2122, 2120)
    


    
![png](enhance_files/enhance_7_1.png)
    


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
    "filter":   "F115W/F277W/F444W",
    "pipeline": "FFT DC-notch + CLAHE clip=0.03",
    "target":   "SN1987A wide field (Program 1232)",
}
with open("enhancement_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved → enhancement_metrics.json")
np.save("enhanced_f444w.npy", enhanced["F444W"])
np.save("enhanced_f277w.npy", enhanced["F277W"])
np.save("enhanced_f115w.npy", enhanced["F115W"])



```

    PSNR : 15.46 dB
    SSIM : 0.2990
    Saved → enhancement_metrics.json
    


```python
print("PSNR/SSIM Interpretation for enhancement tasks:")
print()
print(f"PSNR = {psnr_val:.2f} dB")
print(f"SSIM = {ssim_val:.4f}")
print()
print("These metrics measure SIMILARITY between original and enhanced.")
print("Low values = large change was made; expected for CLAHE.")
print("PSNR/SSIM are most meaningful for denoising tasks where the")
print("goal is to recover an original clean image.")
print()
print("For enhancement the better metric is visual quality + SNR:")

# Signal to Noise Ratio on F277W
sci_f277 = bands["F277W"]["sci"]
sci_f277_clean = fft_gentle(sci_f277)
valid = sci_f277_clean[np.isfinite(sci_f277_clean)]
snr = np.nanmean(valid) / np.nanstd(valid)
print(f"F277W SNR before enhancement: {snr:.2f}")

enh_f277 = enhanced["F277W"]
snr_enh  = enh_f277.mean() / (enh_f277.std() + 1e-10)
print(f"F277W SNR after  enhancement: {snr_enh:.2f}")
print()
print("A higher SNR after enhancement = better for detection tasks.")
print("The visual result (ring clearly visible) is the primary metric.")
```

    PSNR/SSIM Interpretation for enhancement tasks:
    
    PSNR = 15.46 dB
    SSIM = 0.2990
    
    These metrics measure SIMILARITY between original and enhanced.
    Low values = large change was made — expected for CLAHE.
    PSNR/SSIM are most meaningful for denoising tasks where the
    goal is to recover an original clean image.
    
    For enhancement the better metric is visual quality + SNR:
      F277W SNR before enhancement: 0.00
    F277W SNR after  enhancement: 0.14
    
    A higher SNR after enhancement = better for detection tasks.
    The visual result (ring clearly visible) is the primary metric.
    
