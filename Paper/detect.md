## Detecting the Nebula


```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval
from scipy import ndimage
from skimage import filters, feature, morphology, measure, exposure
from skimage.transform import hough_circle, hough_circle_peaks
import cv2


enhanced = {
    "F115W": np.load("enhanced_f115w.npy"),
    "F277W": np.load("enhanced_f277w.npy"),
    "F444W": np.load("enhanced_f444w.npy"),
}

print("Loaded enhanced bands:")
for k, v in enhanced.items():
    print(f"  {k}: shape={v.shape}  range={v.min():.3f} - {v.max():.3f}")

# Work on F444W — ring is brightest here
sci = enhanced["F444W"].copy()
print(f"\nWorking on F444W: {sci.shape}")
```

    Loaded enhanced bands:
      F115W: shape=(2122, 2120)  range=0.000–1.000
      F277W: shape=(2122, 2120)  range=0.000–1.000
      F444W: shape=(2122, 2120)  range=0.000–1.000
    
    Working on F444W: (2122, 2120)
    


```python
# Three classical edge detectors, each with different properties:
#   Sobel     — gradient magnitude, good for smooth edges
#   Canny     — multi-scale, finds thin precise edges, most used in CV
#   Laplacian — second derivative, finds zero crossings at edges

# Normalise to 0-1 float for skimage
img = sci.copy()
img = np.where(np.isfinite(img), img, 0.0)
img = (img - img.min()) / (img.max() - img.min() + 1e-10)

# Sobel
sobel_x    = filters.sobel_h(img)   # horizontal gradient
sobel_y    = filters.sobel_v(img)   # vertical gradient
sobel_mag  = np.hypot(sobel_x, sobel_y)

# Canny 
# sigma controls smoothing before edge detection
# low/high threshold controls which edges are kept
canny_fine  = feature.canny(img, sigma=1.0, low_threshold=0.05, high_threshold=0.15)
canny_coarse= feature.canny(img, sigma=3.0, low_threshold=0.05, high_threshold=0.15)

# Laplacian of Gaussian (LoG)
from scipy.ndimage import gaussian_laplace
log_edges = gaussian_laplace(img, sigma=2.0)
log_edges = np.abs(log_edges)


fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0,0].imshow(img,          origin="lower", cmap="inferno")
axes[0,0].set_title("F444W — enhanced input",  fontsize=11)

axes[0,1].imshow(sobel_mag,    origin="lower", cmap="hot")
axes[0,1].set_title("Sobel — gradient magnitude", fontsize=11)

axes[0,2].imshow(canny_fine,   origin="lower", cmap="gray")
axes[0,2].set_title("Canny σ=1.0 (fine edges)", fontsize=11)

axes[1,0].imshow(canny_coarse, origin="lower", cmap="gray")
axes[1,0].set_title("Canny σ=3.0 (coarse edges)", fontsize=11)

axes[1,1].imshow(log_edges,    origin="lower", cmap="hot")
axes[1,1].set_title("Laplacian of Gaussian σ=2.0", fontsize=11)

# Overlay Canny on original
axes[1,2].imshow(img,          origin="lower", cmap="inferno", alpha=0.8)
axes[1,2].imshow(canny_coarse, origin="lower", cmap="Greens",  alpha=0.5)
axes[1,2].set_title("Canny overlay on F444W", fontsize=11)

for ax in axes.flat:
    ax.axis("off")

plt.suptitle("SN 1987A — Edge detection comparison", fontsize=13)
plt.tight_layout()
plt.savefig("edge_detection.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → edge_detection.png")
```


    
![png](detect_files/detect_2_0.png)
    


    Saved → edge_detection.png
    


```python
# Morphological ops work on binary images (thresholded):
#   Erosion   — shrinks bright regions, removes thin noise
#   Dilation  — expands bright regions, fills gaps
#   Opening   — erosion then dilation — removes small noise blobs
#   Closing   — dilation then erosion — fills holes in sources

# Threshold the Canny edge map to get binary
binary = canny_coarse.astype(np.uint8)

# Structuring element — disk shape, radius 2
disk2 = morphology.disk(2)
disk4 = morphology.disk(4)

eroded  = morphology.erosion(binary,  disk2)
dilated = morphology.dilation(binary, disk2)
opened  = morphology.opening(binary,  disk2)
closed  = morphology.closing(binary,  disk4)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

axes[0,0].imshow(binary,  origin="lower", cmap="gray")
axes[0,0].set_title("Binary (Canny threshold)", fontsize=11)

axes[0,1].imshow(eroded,  origin="lower", cmap="gray")
axes[0,1].set_title("Erosion — removes thin noise", fontsize=11)

axes[0,2].imshow(dilated, origin="lower", cmap="gray")
axes[0,2].set_title("Dilation — expands edges", fontsize=11)

axes[1,0].imshow(opened,  origin="lower", cmap="gray")
axes[1,0].set_title("Opening — removes small blobs", fontsize=11)

axes[1,1].imshow(closed,  origin="lower", cmap="gray")
axes[1,1].set_title("Closing — fills gaps in ring", fontsize=11)

# Overlay closed on original
axes[1,2].imshow(img,    origin="lower", cmap="inferno", alpha=0.8)
axes[1,2].imshow(closed, origin="lower", cmap="Reds",    alpha=0.5)
axes[1,2].set_title("Closed edges overlay", fontsize=11)

for ax in axes.flat:
    ax.axis("off")

plt.suptitle("Morphological operations on edge map", fontsize=13)
plt.tight_layout()
plt.savefig("morphological_ops.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → morphological_ops.png")
```


    
![png](detect_files/detect_3_0.png)
    


    Saved → morphological_ops.png
    


```python
from photutils.detection import DAOStarFinder
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.stats import sigma_clipped_stats

# Load raw F444W for source detection
# (photutils works better on physical flux units than CLAHE-normalised data)
bands_raw = {}
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
    bands_raw[filt.upper()] = {"sci": sci, "wcs": wcs}

sci_raw_f444 = bands_raw["F444W"]["sci"].copy()
sci_raw_f444 = np.where(np.isfinite(sci_raw_f444), sci_raw_f444, 0.0)

# Background stats
mean, median, std = sigma_clipped_stats(
    sci_raw_f444[sci_raw_f444 > 0], sigma=3.0
)
print(f"Background — mean:{mean:.5f}  median:{median:.5f}  std:{std:.5f}")

# DAOStarFinder — detects point sources above threshold
# fwhm: expected star width in pixels
# threshold: detection limit in sigma above background
daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)
sources = daofind(sci_raw_f444 - median)

if sources is None:
    print("No sources found — try lowering threshold")
else:
    print(f"\nDetected {len(sources)} sources")
    print(sources["id","xcentroid","ycentroid","peak","flux"][:10])

# Convert pixel → sky coordinates
wcs_f444 = bands_raw["F444W"]["wcs"]
if sources is not None:
    xs = np.array(sources["xcentroid"])
    ys = np.array(sources["ycentroid"])

    # Resize coords to match resized image if needed
    orig_shape = bands_raw["F444W"]["sci"].shape
    tgt_shape  = enhanced["F444W"].shape
    if orig_shape != tgt_shape:
        xs = xs * (tgt_shape[1] / orig_shape[1])
        ys = ys * (tgt_shape[0] / orig_shape[0])

    sky = wcs_f444.pixel_to_world(xs, ys)
    sources["ra"]  = sky.ra.deg
    sources["dec"] = sky.dec.deg

    # Plot sources on enhanced image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(enhanced["F444W"], origin="lower", cmap="inferno")
    ax.scatter(xs, ys, s=30, facecolors="none",
               edgecolors="cyan", linewidths=0.8, label=f"{len(sources)} sources")
    ax.set_title(f"F444W — {len(sources)} sources detected (DAOStarFinder)", fontsize=11)
    ax.legend(fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("source_detection.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → source_detection.png")

    # Save catalog
    sources.write("source_catalog.csv",
                  format="csv", overwrite=True)
    print(f"Saved → source_catalog.csv ({len(sources)} sources)")
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
    WARNING: FITSFixedWarning: 'datfix' made the change 'Set DATE-BEG to '2022-09-02T13:57:06.836' from MJD-BEG.
    Set DATE-AVG to '2022-09-02T14:12:58.844' from MJD-AVG.
    Set DATE-END to '2022-09-02T14:28:48.449' from MJD-END'. [astropy.wcs.wcs]
    WARNING: FITSFixedWarning: 'obsfix' made the change 'Set OBSGEO-L to   -46.668572 from OBSGEO-[XYZ].
    Set OBSGEO-B to   -16.348305 from OBSGEO-[XYZ].
    Set OBSGEO-H to 1478590470.575 from OBSGEO-[XYZ]'. [astropy.wcs.wcs]
    

    Background — mean:0.03927  median:0.02851  std:0.03650
    
    Detected 181 sources
     id     x_centroid         y_centroid        peak       flux   
    --- ------------------ ------------------ ---------- ----------
      1  89.31547757202743 0.6199571263770884 0.18394196   1.081608
      2 202.98308439128027 0.6656767444834635  0.4978995  2.9546444
      3  232.9967515179337 0.5408106624225659  0.1976725  1.1575012
      4  175.5550063620294 2.8856446530766435 0.27539638 0.84442365
      5 286.78465708486726  4.834995202819179 0.36837444   5.568093
      6  78.14066780489702  7.322636022581371 0.94045776  7.4320884
      7 137.18299374015686  8.403775483777276 0.58174676  4.7702885
      8 254.84235697541504  9.272023675570063 0.40753463  2.2872527
      9  25.83646965298854 10.269792045413547  0.6992216  5.8658404
     10 121.07618648072231 10.418112967358013  27.325407  224.71758
    

    WARNING: AstropyDeprecationWarning: The column name 'xcentroid' was deprecated in version 3.0. Use 'x_centroid' instead. It will be removed in version 4.0. Once you have updated your code to use 'x_centroid', set photutils.future_column_names = True to opt into a standard QTable without the deprecated column name mapping. [photutils.utils._deprecation]
    WARNING: AstropyDeprecationWarning: The column name 'ycentroid' was deprecated in version 3.0. Use 'y_centroid' instead. It will be removed in version 4.0. Once you have updated your code to use 'y_centroid', set photutils.future_column_names = True to opt into a standard QTable without the deprecated column name mapping. [photutils.utils._deprecation]
    


    
![png](detect_files/detect_4_3.png)
    


    Saved → source_detection.png
    Saved → source_catalog.csv (181 sources)
    


```python
from skimage.transform import hough_circle, hough_circle_peaks
from skimage import feature
from astropy.io import fits
from astropy.wcs import WCS
from pathlib import Path
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt

# Load
ring_path = Path("./jwst_data/sn1987a") / \
            "jw01726-o001_t001_nircam_clear-f444w-sub320_i2d.fits"

with fits.open(ring_path) as hdul:
    ring_sci = hdul["SCI"].data.astype(np.float32)
    ring_wcs = WCS(hdul["SCI"].header)

ring_sci  = np.where(np.isfinite(ring_sci), ring_sci, 0.0)
vmin      = np.percentile(ring_sci[ring_sci > 0], 2)
vmax      = np.percentile(ring_sci[ring_sci > 0], 99)
ring_norm = np.clip((ring_sci - vmin) / (vmax - vmin + 1e-10), 0, 1)


smoothed         = gaussian_filter(ring_norm, sigma=10)
ring_py, ring_px = np.unravel_index(np.argmax(smoothed), smoothed.shape)
print(f"Ring centre estimate: x={ring_px}, y={ring_py}")

# Crop 80px window around ring
half = 80
x0   = max(0, ring_px - half)
x1   = min(ring_norm.shape[1], ring_px + half)
y0   = max(0, ring_py - half)
y1   = min(ring_norm.shape[0], ring_py + half)
crop = ring_norm[y0:y1, x0:x1]
print(f"Crop shape: {crop.shape}")

# Canny on crop
edges = feature.canny(crop, sigma=1.0,
                      low_threshold=0.02, high_threshold=0.1)
print(f"Edge pixels in crop: {edges.sum()}")

# HOUGH
radii     = np.arange(15, 55, 1)
hough_res = hough_circle(edges, radii)

accums, cx_arr, cy_arr, radii_found = hough_circle_peaks(
    hough_res, radii,
    num_peaks=1,
    min_xdistance=20,
    min_ydistance=20,
)

pixel_scale = float(np.abs(ring_wcs.pixel_scale_matrix[0, 0])) * 3600
print(f"\nBest detection:")
for acc, cx, cy, r in zip(accums, cx_arr, cy_arr, radii_found):
    r_as = r * pixel_scale
    print(f"  centre crop=({cx},{cy})  full=({cx+x0},{cy+y0})")
    print(f"  radius = {r}px = {r_as:.3f} arcsec")
    print(f"  literature = 0.85 arcsec  |  difference = {abs(r_as-0.85):.3f} arcsec")
    print(f"  Hough score = {acc:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Full image with crop box
axes[0].imshow(ring_norm, origin="lower", cmap="inferno")
axes[0].add_patch(plt.Rectangle((x0,y0), x1-x0, y1-y0,
                  edgecolor="yellow", linewidth=1.5, fill=False))
axes[0].set_title("Full field — yellow = search region", fontsize=11)
axes[0].axis("off")

# Crop + edges overlay
axes[1].imshow(crop,  origin="lower", cmap="inferno", alpha=0.8)
axes[1].imshow(edges, origin="lower", cmap="Greens",  alpha=0.5)
axes[1].set_title(f"Ring crop + Canny edges ({edges.sum()} px)", fontsize=11)
axes[1].axis("off")

# Hough result
axes[2].imshow(crop, origin="lower", cmap="inferno")
for acc, cx, cy, r in zip(accums, cx_arr, cy_arr, radii_found):
    r_as = r * pixel_scale
    axes[2].add_patch(plt.Circle((cx, cy), r,
                                  color="yellow", fill=False,
                                  linewidth=2.5,
                                  label=f"r={r}px = {r_as:.2f}\""))
    axes[2].scatter([cx], [cy], c="yellow", s=80, marker="+", linewidths=2)
axes[2].legend(fontsize=10, loc="lower right")
axes[2].set_title("Best Hough circle", fontsize=11)
axes[2].axis("off")

plt.suptitle("SN1987A — Hough circle transform (focused)", fontsize=13)
plt.tight_layout()
plt.savefig("hough_circle.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → hough_circle.png")
```

    WARNING: FITSFixedWarning: 'datfix' made the change 'Set DATE-BEG to '2022-09-02T13:57:06.836' from MJD-BEG.
    Set DATE-AVG to '2022-09-02T14:12:58.844' from MJD-AVG.
    Set DATE-END to '2022-09-02T14:28:48.449' from MJD-END'. [astropy.wcs.wcs]
    WARNING: FITSFixedWarning: 'obsfix' made the change 'Set OBSGEO-L to   -46.668572 from OBSGEO-[XYZ].
    Set OBSGEO-B to   -16.348305 from OBSGEO-[XYZ].
    Set OBSGEO-H to 1478590470.575 from OBSGEO-[XYZ]'. [astropy.wcs.wcs]
    

    Ring centre estimate: x=262, y=259
    Crop shape: (147, 143)
    Edge pixels in crop: 634
    
    Best detection:
      centre crop=(80,84)  full=(262,263)
      radius = 15px = 0.256 arcsec
      literature = 0.85 arcsec  |  difference = 0.594 arcsec
      Hough score = 0.500
      centre crop=(102,85)  full=(284,264)
      radius = 29px = 0.494 arcsec
      literature = 0.85 arcsec  |  difference = 0.356 arcsec
      Hough score = 0.196
      centre crop=(62,56)  full=(244,235)
      radius = 37px = 0.630 arcsec
      literature = 0.85 arcsec  |  difference = 0.220 arcsec
      Hough score = 0.148
      centre crop=(41,67)  full=(223,246)
      radius = 35px = 0.596 arcsec
      literature = 0.85 arcsec  |  difference = 0.254 arcsec
      Hough score = 0.140
      centre crop=(97,110)  full=(279,289)
      radius = 54px = 0.920 arcsec
      literature = 0.85 arcsec  |  difference = 0.070 arcsec
      Hough score = 0.115
    


    
![png](detect_files/detect_5_2.png)
    


    Saved → hough_circle.png
    


```python

from sklearn.decomposition import PCA
import json

all_bands = {}
for fname, arr in enhanced.items():
    all_bands[fname] = arr

for f in sorted(Path("./jwst_data/sn1987a").glob("*sub320*i2d.fits")):
    with fits.open(f) as hdul:
        sci  = hdul["SCI"].data.astype(np.float32)
        phdr = hdul[0].header
        filt = phdr.get("FILTER", "UNKNOWN")
    sci = np.where(np.isfinite(sci), sci, 0.0)
    all_bands[f"sub_{filt}"] = sci

print("Bands for PCA:")
for k, v in all_bands.items():
    print(f"  {k}: {v.shape}")

# Resize all to smallest shape to avoid memory issues
min_h = min(v.shape[0] for v in all_bands.values())
min_w = min(v.shape[1] for v in all_bands.values())
print(f"\nResizing all to: {min_h}x{min_w}")

cube = []
band_names = []
for fname, arr in all_bands.items():
    resized = cv2.resize(arr, (min_w, min_h), interpolation=cv2.INTER_AREA)
    cube.append(resized.ravel())
    band_names.append(fname)

# Shape: (n_bands, n_pixels) → transpose to (n_pixels, n_bands)
X = np.array(cube).T
print(f"Data matrix shape: {X.shape}")

# PCA
n_components = min(6, len(band_names))
pca  = PCA(n_components=n_components)
pca.fit(X)
Xt   = pca.transform(X)

print(f"\nExplained variance ratio:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var*100:.1f}%")
print(f"  Total: {sum(pca.explained_variance_ratio_)*100:.1f}%")

# Reshape components back to images
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for i, ax in enumerate(axes.flat):
    if i >= n_components:
        ax.axis("off")
        continue
    comp = Xt[:, i].reshape(min_h, min_w)
    vabs = np.percentile(np.abs(comp), 99)
    ax.imshow(comp, origin="lower", cmap="RdBu_r", vmin=-vabs, vmax=vabs)
    var_pct = pca.explained_variance_ratio_[i] * 100
    ax.set_title(f"PC{i+1} — {var_pct:.1f}% variance", fontsize=11)
    ax.axis("off")

plt.suptitle("PCA decomposition — SN1987A multi-band", fontsize=13)
plt.tight_layout()
plt.savefig("pca_components.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → pca_components.png")


pca_results = {
    "bands": band_names,
    "explained_variance_pct": [float(round(v*100, 2))
                                for v in pca.explained_variance_ratio_],
    "total_variance_pct": float(round(sum(pca.explained_variance_ratio_)*100, 2)),
}
with open("pca_results.json", "w") as f:
    json.dump(pca_results, f, indent=2)
print("Saved → pca_results.json")
```

    Bands for PCA:
      F115W: (2122, 2120)
      F277W: (2122, 2120)
      F444W: (2122, 2120)
      sub_F150W: (795, 810)
      sub_F200W: (794, 809)
      sub_F444W: (326, 325)
    
    Resizing all to: 326x325
    Data matrix shape: (105950, 6)
    
    Explained variance ratio:
      PC1: 79.7%
      PC2: 10.7%
      PC3: 9.6%
      PC4: 0.0%
      PC5: 0.0%
      PC6: 0.0%
      Total: 100.0%
    


    
![png](detect_files/detect_6_1.png)
    


    Saved → pca_components.png
    Saved → pca_results.json
    


```python
from pathlib import Path
from astropy.io import fits
from skimage import feature
import numpy as np

# Find the file
print("Searching for sub320 files:")
for f in Path(".").rglob("*sub320*i2d.fits"):
    print(f"  {f}")

print()

print("bands_ring keys:", list(bands_raw.keys()))
print("bands_ring F444W shape:", bands_raw["F444W"]["sci"].shape)

# edge detection directly on bands_ring F444W
sci = bands_raw["F444W"]["sci"].copy()
sci = np.where(np.isfinite(sci), sci, 0.0)
vmin = np.percentile(sci[sci > 0], 2)
vmax = np.percentile(sci[sci > 0], 99)
norm = np.clip((sci - vmin) / (vmax - vmin + 1e-10), 0, 1)

print(f"\nNormalised range: {norm.min():.3f} – {norm.max():.3f}")
print(f"Non-zero pixels: {(norm > 0).sum()}")

# multiple Canny settings
for sigma, lo, hi in [(0.5, 0.01, 0.05),
                       (1.0, 0.01, 0.05),
                       (1.0, 0.02, 0.08),
                       (2.0, 0.05, 0.15)]:
    edges = feature.canny(norm, sigma=sigma, 
                          low_threshold=lo, high_threshold=hi)
    print(f"  sigma={sigma} lo={lo} hi={hi} → {edges.sum()} edge pixels")
```

    Searching for sub320 files:
      jwst_data\sn1987a\jw01726-o001_t001_nircam_clear-f150w-sub320_i2d.fits
      jwst_data\sn1987a\jw01726-o001_t001_nircam_clear-f200w-sub320_i2d.fits
      jwst_data\sn1987a\jw01726-o001_t001_nircam_clear-f444w-sub320_i2d.fits
      jwst_data\sn1987a\MAST_2026-04-28T0559\JWST\jw01726-o001_t001_nircam_clear-f150w-sub320_i2d.fits
      jwst_data\sn1987a\MAST_2026-04-28T0559\JWST\jw01726-o001_t001_nircam_clear-f200w-sub320_i2d.fits
      jwst_data\sn1987a\MAST_2026-04-28T0559\JWST\jw01726-o001_t001_nircam_clear-f212n-sub320_i2d.fits
      jwst_data\sn1987a\MAST_2026-04-28T0559\JWST\jw01726-o001_t001_nircam_clear-f356w-sub320_i2d.fits
      jwst_data\sn1987a\MAST_2026-04-28T0559\JWST\jw01726-o001_t001_nircam_clear-f444w-sub320_i2d.fits
      jwst_data\sn1987a\MAST_2026-04-28T0559\JWST\jw01726-o001_t001_nircam_f150w2-f164n-sub320_i2d.fits
      jwst_data\sn1987a\MAST_2026-04-28T0559\JWST\jw01726-o001_t001_nircam_f322w2-f323n-sub320_i2d.fits
      jwst_data\sn1987a\MAST_2026-04-28T0559\JWST\jw01726-o001_t001_nircam_f405n-f444w-sub320_i2d.fits
    
    bands_ring keys: ['F115W', 'F277W', 'F444W', 'F150W', 'F200W']
    bands_ring F444W shape: (326, 325)
    
    Normalised range: 0.000 – 1.000
    Non-zero pixels: 69372
      sigma=0.5 lo=0.01 hi=0.05 → 5411 edge pixels
      sigma=1.0 lo=0.01 hi=0.05 → 3477 edge pixels
      sigma=1.0 lo=0.02 hi=0.08 → 2158 edge pixels
      sigma=2.0 lo=0.05 hi=0.15 → 600 edge pixels
    
