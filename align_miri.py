import numpy as np
from astropy.io import fits
from align_and_merge import align_and_merge, plot_alignment_result

# 1. Update these strings with your actual filenames
miri_files = {
    "F560W":  "Data/jw01232-o002_t001_miri_f560w_i2d.fits",
    "F1000W": "Data/jw01232-o002_t001_miri_f1000w_i2d.fits",
    "F2550W": "Data/jw01232-o001_t001_miri_f2550w-brightsky_i2d.fits"
}

# 2. Load the data into the format expected by the script
bands_miri = {}
for filter_name, path in miri_files.items():
    with fits.open(path) as hdul:
        bands_miri[filter_name] = {
            "sci": hdul["SCI"].data.astype(np.float32),
            "path": path
        }

# 3. Process the merge (using WCS for instrument-to-instrument accuracy)
miri_result = align_and_merge(
    bands_miri,
    blue_filter  = "F560W",
    green_filter = "F1000W",
    red_filter   = "F2550W",
    use_orb      = False 
)

# 4. Save the result
plot_alignment_result(miri_result)