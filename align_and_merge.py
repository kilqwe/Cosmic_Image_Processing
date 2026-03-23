import cv2
import numpy as np
import pywt
from pathlib import Path
from astropy.visualization import ZScaleInterval
from astropy.io import fits
from astropy.wcs import WCS
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt


#  Normalise float32 SCI → uint8 for feature detectors (like ORB) that expect 8-bit input.

def to_uint8(sci: np.ndarray):
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(sci[np.isfinite(sci)])
    clipped = np.clip(sci, vmin, vmax)
    scaled  = (clipped - vmin) / (vmax - vmin)
    return (scaled * 255).astype(np.uint8)


# ORB feature-based alignment

def align_orb(reference: np.ndarray, moving: np.ndarray,max_features: int = 5000,match_frac: float = 0.15):
    ref_u8  = to_uint8(reference)
    mov_u8  = to_uint8(moving)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(ref_u8, None)
    kp2, des2 = orb.detectAndCompute(mov_u8, None)

    print(f"Keypoints — reference: {len(kp1)}  moving: {len(kp2)}")

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("Not enough keypoints — falling back to no alignment")
        return moving.copy(), None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)

    n_keep  = max(4, int(len(matches) * match_frac))
    good = matches[:n_keep]
    print(f" Matches: {len(matches)} total → {len(good)} kept")

    if len(good) < 4:
        print("Too few good matches — skipping alignment")
        return moving.copy(), None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    inliers = mask.ravel().sum()
    print(f"Homography inliers: {inliers} / {len(good)}")

    if H is None or inliers < 4:
        print("Homography failed — skipping alignment")
        return moving.copy(), None

    h, w = reference.shape
    mov_f  = moving.astype(np.float32)
    mov_f[~np.isfinite(mov_f)] = 0.0
    warped = cv2.warpPerspective(mov_f, H, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=np.nan)
    return warped, H


# WCS-based alignment (more accurate we'll use as comparison in report)

def align_wcs(reference_path: Path, moving_path: Path):
    try:
        from reproject import reproject_interp
    except ImportError:
        print("  reproject not installed: pip install reproject")
        return None

    with fits.open(reference_path) as ref_hdul:
        ref_header = ref_hdul["SCI"].header
        ref_shape  = ref_hdul["SCI"].data.shape
    with fits.open(moving_path) as mov_hdul:
        mov_data   = mov_hdul["SCI"].data.astype(np.float32)
        mov_header = mov_hdul["SCI"].header

    warped, _ = reproject_interp(
        (mov_data, mov_header),
        ref_header,
        shape_out=ref_shape,
    )
    return warped.astype(np.float32)


#  Wavelet combination

def wavelet_merge(bands: list[np.ndarray], wavelet: str = "db4",level: int = 4, mode: str = "max"):
    clean = [np.where(np.isfinite(b), b, 0.0) for b in bands]
    coeffs_list = [pywt.wavedec2(b, wavelet=wavelet, level=level) for b in clean]

    merged_coeffs = []
    for level_idx in range(len(coeffs_list[0])):
        if level_idx == 0:
            approx = np.mean([c[0] for c in coeffs_list], axis=0)
            merged_coeffs.append(approx)
        else:
            fused_details = []
            for subband_idx in range(3):
                stack = np.stack([c[level_idx][subband_idx]
                                  for c in coeffs_list], axis=0)
                if mode == "max":
                    idx   = np.argmax(np.abs(stack), axis=0)
                    fused = np.take_along_axis(stack, idx[np.newaxis], axis=0)[0]
                else:
                    fused = np.mean(stack, axis=0)
                fused_details.append(fused)
            merged_coeffs.append(tuple(fused_details))

    merged = pywt.waverec2(merged_coeffs, wavelet=wavelet)
    h, w   = bands[0].shape
    return merged[:h, :w].astype(np.float32)


# Full pipeline -  align bands → wavelet merge → return RGB + fused arrays.

def align_and_merge(bands: dict, blue_filter:  str = "F115W",green_filter: str = "F277W",red_filter:   str = "F444W",use_orb:      bool = True) :

    ref_key = red_filter
    ref     = bands[ref_key]["sci"]
    aligned = {ref_key: ref}

    for key in [blue_filter, green_filter]:
        if key not in bands:
            print(f"  Warning: {key} not in bands — skipping")
            continue
        print(f"\nAligning {key} → {ref_key}...")
        sci = bands[key]["sci"]

        if use_orb:
            warped, H = align_orb(ref, sci)
            status = "ORB OK" if H is not None else "ORB failed — using as-is"
            print(f"  {status}")
        else:
            warped = align_wcs(bands[ref_key]["path"], bands[key]["path"])
            if warped is None:
                warped = sci

        aligned[key] = warped

    print("\nWavelet merging...")
    to_merge = [aligned.get(k, ref) for k in [blue_filter, green_filter, red_filter]]
    merged   = wavelet_merge(to_merge, wavelet="db4", level=4, mode="max")
    print(f"Merged shape: {merged.shape}")

    return {
        "blue":   aligned.get(blue_filter, ref),
        "green":  aligned.get(green_filter, ref),
        "red":    ref,
        "merged": merged,
    }


# Plot results (for report) — shows aligned RGB + wavelet-fused sharpness layer.

def plot_alignment_result(result: dict, save: bool = True):
    from astropy.visualization import ZScaleInterval, ImageNormalize, LinearStretch

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    titles = ["Blue (aligned)", "Green (aligned)", "Red (reference)", "Wavelet merged"]
    keys   = ["blue", "green", "red", "merged"]

    for ax, key, title in zip(axes, keys, titles):
        data = result[key]
        norm = ImageNormalize(data[np.isfinite(data)],
                              interval=ZScaleInterval(),
                              stretch=LinearStretch())
        ax.imshow(data, origin="lower", cmap="inferno", norm=norm)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    if save:
        plt.savefig("alignment_result.png", dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

