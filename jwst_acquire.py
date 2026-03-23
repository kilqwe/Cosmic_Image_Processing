import argparse
import json
from pathlib import Path
import astropy.table
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval, ImageNormalize, LinearStretch
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations


TARGETS = {
    "carina": {
        "name":        "Cosmic Cliffs — Carina Nebula",
        "coord":       SkyCoord("10h44m57s -59d52m00s", frame="icrs"),
        "program_id":  "2731",
        "radius_deg":  0.05,
        "instrument":  "NIRCAM/IMAGE",
        "rgb_filters": ["f090w", "f187n", "f444w"],   # blue, green, red
        "notes":       "First JWST release image. Rich in fine structure.",
    },

    "sn1987a": {
        "name":        "SN 1987A — active supernova",
        "coord":       SkyCoord("05h35m28s -69d16m11s", frame="icrs"),
        "program_id":  "1726",
        "radius_deg":  0.01,                           # small — compact source
        "instrument":  "NIRCAM/IMAGE",
        "rgb_filters": ["f162m", "f277w", "f444w"],
        "notes":       "Multi-epoch — great for temporal change detection.",
    },

    "ceers": {
        "name":        "CEERS Deep Field",
        "coord":       SkyCoord("214.825 +52.825", unit="deg", frame="icrs"),
        "program_id":  "1345",
        "radius_deg":  0.1,
        "instrument":  "NIRCAM/IMAGE",
        "rgb_filters": ["f115w", "f200w", "f444w"],
        "notes":       "Thousands of galaxies. Best for ML classification at scale.",
    },

# when we want to use some other dataset not in the above list, we can use the "custom" key and fill in the details as needed.
    "custom": {
        "name":        None,
        "coord":       None,
        "program_id":  None,
        "radius_deg":  0.05,
        "instrument":  "NIRCAM/IMAGE",
        "rgb_filters": ["f090w", "f200w", "f444w"],   # safe fallback
        "notes":       "User-supplied target.",
    },
}

BASE_DIR = Path("./jwst_data")


def search(target: dict, verbose: bool = True):

    print(f"Searching: {target['name']}")
    print(f"Coord: {target['coord'].ra.deg:.5f}°  {target['coord'].dec.deg:.5f}°")
    print(f"Radius   : {target['radius_deg']}°")


    obs = Observations.query_criteria(
        coordinates=target["coord"],
        radius=target["radius_deg"] * u.deg,
        obs_collection="JWST",
        instrument_name=target["instrument"],
        calib_level=3,
    )

    if len(obs) == 0:
        print("  No observations found. Try increasing radius_deg or check the coordinates.")
        return obs

    if verbose:
        print(f"  Found {len(obs)} observations\n")
        print(obs["obs_id", "target_name", "filters", "t_exptime", "t_obs_release"][:8])

    return obs


def _resolve_paths(manifest, out_dir: str = "./jwst_data") :
  
    resolved = []
    for raw in manifest["Local Path"]:
        p = Path(raw)
        if p.exists():
            resolved.append(p)
            continue
        # Search the entire out_dir tree for the filename
        fname = p.name
        if not fname.endswith(".fits"):
            continue
        matches = list(Path(out_dir).rglob(fname))
        if matches:
            resolved.append(matches[0])
        else:
            print(f"  Warning: could not locate {fname} under {out_dir}")
    return resolved


DETECTOR_NAMES = ["nrca1","nrca2","nrca3","nrca4","nrcb1","nrcb2","nrcb3","nrcb4"]

def inspect_products(obs):


    products = Observations.get_product_list(obs)
    i2d = Observations.filter_products(
        products,
        productSubGroupDescription="I2D",
        extension="fits",
    )

    print(f"\n  {'#':<4} {'filename':<65} {'type'}")
    print(f"  {'─'*4} {'─'*65} {'─'*14}")

    has_mosaic  = False
    has_chips   = False

    for i, row in enumerate(i2d):
        name = row["productFilename"]
        size_mb = (row["size"] or 0) / 1e6
        is_chip = any(d in name for d in DETECTOR_NAMES)

        if is_chip:
            kind = "chip (detector)"
            has_chips = True
        else:
            kind = "MOSAIC ← use this"
            has_mosaic = True

        size_str = f"{size_mb:.0f} MB" if size_mb > 0 else "size N/A"
        print(f"  [{i:<2}] {name:<65} {kind}  {size_str}")

    print()
    if has_mosaic:
        print("  Filter-named mosaics found — use download() normally.")
    elif has_chips:
        print("  Only detector chips found — use download_chips() or search_by_program().")
    else:
        print("  No i2d files found at all — try search_by_program().")

    return i2d


def search_by_program(program_id: str):
    print(f"\n  Searching by program ID: {program_id}")
    obs = Observations.query_criteria(
        obs_collection="JWST",
        proposal_id=program_id,
        calib_level=3,
        instrument_name="NIRCAM/IMAGE",
    )
    print(f"  Found {len(obs)} observations for program {program_id}")
    return inspect_products(obs)


def download_by_index(i2d_table, indices: list[int], out_dir: str = "./jwst_data"):

    subset = i2d_table[indices]

    print(f"\n  Downloading {len(subset)} file(s):")
    for row in subset:
        size_mb = (row["size"] or 0) / 1e6
        print(f"    {row['productFilename']}  ({size_mb:.0f} MB)")

    manifest = Observations.download_products(subset, download_dir=out_dir)
    paths = _resolve_paths(manifest, out_dir)
    print(f"\n  Saved {len(paths)} file(s) to: {out_dir}")
    return paths




def download_chips(i2d_table, preferred_detector: str = "nrca1",
                   out_dir: str = "./jwst_data") -> list[Path]:

    names   = np.array([r["productFilename"] for r in i2d_table])
    is_chip = np.array([preferred_detector in n for n in names])
    subset  = i2d_table[is_chip]

    if len(subset) == 0:
        print(f"  No files found for detector '{preferred_detector}'.")
        print(f"  Available detectors: {DETECTOR_NAMES}")
        return []

    print(f"\n  Found {len(subset)} '{preferred_detector}' chip file(s):")
    for row in subset:
        print(f"    {row['productFilename']}")

    manifest = Observations.download_products(subset, download_dir=out_dir)
    paths = _resolve_paths(manifest, out_dir)

    print("\n  Filter check (from FITS headers):")
    for p in paths:
        with fits.open(p) as hdul:
            phdr   = hdul[0].header
            filt   = phdr.get("FILTER", "?")
            pupil  = phdr.get("PUPIL",  "?")
            tgt    = phdr.get("TARGNAME", "?")
        print(f" {p.name}  ->  FILTER={filt}  PUPIL={pupil}  TARGET={tgt}")

    return paths


def download(obs, target: dict, dry_run: bool = False):
    if len(obs) == 0:
        return []

    products = Observations.get_product_list(obs)
    filtered = Observations.filter_products(
        products,
        productSubGroupDescription="I2D",
        extension="fits",
    )

    rgb  = target["rgb_filters"]
    mask = np.zeros(len(filtered), dtype=bool)
    for f in rgb:
        mask |= np.array([f in n.lower() for n in filtered["productFilename"]])

    subset = filtered[mask]

    if len(subset) == 0:
        print(f"No i2d products matched filters {rgb}.")
        print(" Run inspect_products(obs) to see all available files.")
        print(" If you see chip files, use download_chips() instead.")
        return []

    print(f"\n  {len(subset)} files to download ({', '.join(rgb)}):")
    for row in subset:
        size_mb = (row["size"] or 0) / 1e6
        print(f"    {row['productFilename']}  ({size_mb:.0f} MB)")

    if dry_run:
        print("\n[dry_run] Skipping download.")
        return []

    out_dir = BASE_DIR / target["name"].split("—")[0].strip().replace(" ", "_").lower()
    manifest = Observations.download_products(subset, download_dir=str(out_dir))

    paths = _resolve_paths(manifest, str(out_dir))
    print(f"\n  Downloaded {len(paths)} files to {out_dir}")
    return paths

def load_fits(path: Path) -> dict:

    with fits.open(path) as hdul:
        sci    = hdul["SCI"].data.astype(np.float32)
        err    = hdul["ERR"].data.astype(np.float32) if "ERR" in hdul else np.zeros(hdul["SCI"].data.shape, dtype=np.float32)
        dq     = hdul["DQ"].data                     if "DQ"  in hdul else np.zeros(hdul["SCI"].data.shape, dtype=np.uint32)
        header = hdul["SCI"].header
        wcs    = WCS(header)

        # Pull metadata from primary header if available
        phdr       = hdul[0].header
        filt       = phdr.get("FILTER", header.get("FILTER", "UNKNOWN"))
        targ_name  = phdr.get("TARGNAME", "unknown")
        exptime    = phdr.get("XPOSURE", phdr.get("EFFEXPTM", float("nan")))

    # Mask bad pixels (DQ != 0) and non-finite values
    bad = (dq != 0) | ~np.isfinite(sci)
    sci[bad] = np.nan
    err[bad] = np.nan

    valid = sci[np.isfinite(sci)]
    print(f"{path.name}")
    print(f"Filter  : {filt}")
    print(f"Shape   : {sci.shape}")
    print(f"Exptime : {exptime:.1f}s")
    print(f"Range   : {np.nanmin(sci):.4f} – {np.nanmax(sci):.4f} MJy/sr")
    print(f"Bad px  : {bad.sum():,} / {sci.size:,} "
          f"({100*bad.sum()/sci.size:.1f}%)")

    return {
        "sci":sci,
        "err": err,
        "wcs":wcs,
        "header":header,
        "path": path,
        "filter": filt.upper(),
        "target":targ_name,
        "exptime":exptime,
    }


def load_bands(fits_paths: list[Path], rgb_filters: list[str] = None):

    bands = {}
    for fpath in fits_paths:
        name = fpath.name.lower()
        is_chip = any(d in name for d in DETECTOR_NAMES)

        if is_chip:
            with fits.open(fpath) as hdul:
                phdr = hdul[0].header
                filt = phdr.get("FILTER", phdr.get("PUPIL", "UNKNOWN")).upper()
            print(f"\nLoading chip {fpath.name}  ->  filter={filt}")
            bands[filt] = load_fits(fpath)
        else:
            if rgb_filters is None:
                continue
            for f in rgb_filters:
                if f in name:
                    print(f"\nLoading {f.upper()}...")
                    bands[f.upper()] = load_fits(fpath)
                    break

    if rgb_filters:
        missing = [f.upper() for f in rgb_filters if f.upper() not in bands]
        if missing:
            print(f"\n  Warning: missing bands {missing} — check downloaded files")

    return bands


def plot_bands(bands: dict, save: bool = True):

    n = len(bands)
    if n == 0:
        print("No bands to plot.")
        return

    fig = plt.figure(figsize=(6 * n, 6))
    gs  = gridspec.GridSpec(1, n, figure=fig, wspace=0.05)

    for i, (fname, data) in enumerate(sorted(bands.items())):
        ax   = fig.add_subplot(gs[i])
        sci  = data["sci"]
        norm = ImageNormalize(sci, interval=ZScaleInterval(), stretch=LinearStretch())
        ax.imshow(sci, origin="lower", cmap="inferno", norm=norm)
        ax.set_title(f"{fname}  ({data['exptime']:.0f}s)", fontsize=11)
        ax.set_xlabel("x (px)")
        if i == 0:
            ax.set_ylabel("y (px)")
        else:
            ax.set_yticklabels([])

    target_label = list(bands.values())[0]["target"].replace(" ", "_")
    fig.suptitle(f"JWST NIRCam — {target_label}", fontsize=13, y=1.01)
    plt.tight_layout()

    if save:
        out = f"{target_label}_quicklook.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved quicklook → {out}")
    else:
        plt.show()

    plt.close()




def summarise(bands: dict) -> dict:
    summary = {}
    for fname, data in sorted(bands.items()):
        sci = data["sci"]
        entry = {
            "filter": fname,
            "shape":  list(sci.shape),
            "exptime": data["exptime"],
            "median": float(np.nanmedian(sci)),
            "std":  float(np.nanstd(sci)),
            "bad_frac": float(np.isnan(sci).mean()),
        }
        summary[fname] = entry
        print(f"  {fname:8s}  shape={sci.shape}  "
              f"median={entry['median']:.5f}  std={entry['std']:.5f}  "
              f"bad={entry['bad_frac']*100:.1f}%")

    return summary


def new_dataset(name: str, ra: float, dec: float,
                rgb_filters: list[str] = None,
                radius_deg: float = 0.05) -> dict:

    target = {
        "name": name,
        "coord": SkyCoord(ra, dec, unit="deg", frame="icrs"),
        "program_id": None,
        "radius_deg":radius_deg,
        "instrument":"NIRCAM/IMAGE",
        "rgb_filters": rgb_filters or ["f090w", "f200w", "f444w"],
        "notes": "Custom target.",
    }

    obs = search(target)
    paths = download(obs, target)

    if not paths:
        print("No files downloaded — returning empty dict")
        return {}

    bands = load_bands(paths, target["rgb_filters"])
    plot_bands(bands)
    return bands



def run(target_key: str, ra=None, dec=None, custom_name=None, dry_run=False):
    if target_key not in TARGETS:
        print(f"Unknown target '{target_key}'. Choose from: {list(TARGETS.keys())}")
        return

    target = dict(TARGETS[target_key])   # copy so we don't mutate the registry

    if target_key == "custom":
        if ra is None or dec is None or custom_name is None:
            ra = float(input("RA  (degrees): "))
            dec = float(input("Dec (degrees): "))
            custom_name = input("Target name : ")
        target["name"]  = custom_name
        target["coord"] = SkyCoord(ra, dec, unit="deg", frame="icrs")

    obs   = search(target)
    paths = download(obs, target, dry_run=dry_run)

    if dry_run or not paths:
        return
    print("  Loading bands...")
    print(f"{'─'*60}")
    bands = load_bands(paths, target["rgb_filters"])

    print(f"\n{'─'*60}")
    print("  Dataset summary")
 
    summary = summarise(bands)

    # Save summary as JSON next to the data
    out_dir = BASE_DIR / target["name"].split("—")[0].strip().replace(" ", "_").lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_bands(bands)

    print(f"\n bands dict keys : {list(bands.keys())}")
    print(" Each value: { sci, err, wcs, header, filter, target, exptime }")
    print("\n Pass `bands` to:")
    print("enhance.py  → Person A (false-color + super-resolution)")
    print("detect.py   → Person B (source detection + classification)")

    return bands




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JWST data acquisition")
    parser.add_argument("--target",  default=None,
                        choices=list(TARGETS.keys()),
                        help="Target key (omit for interactive menu)")
    parser.add_argument("--ra",type=float,help="RA in degrees (custom only)")
    parser.add_argument("--dec", type=float,help="Dec in degrees (custom only)")
    parser.add_argument("--name",type=str,help="Name for custom target")
    parser.add_argument("--dry-run", action="store_true",
                        help="Search only, skip download")
    args = parser.parse_args()

    key = args.target

#  Finding new JWST releases:
#    https://mast.stsci.edu  -> Mission=JWST, Calib Level=3, sort by Release Date
#    https://www.stsci.edu/jwst/science-execution/approved-programs/general-observers