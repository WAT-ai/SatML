import os
import numpy as np
import rasterio
from importlib import import_module
import sys
from image_utils import data_generator
import spectral.io.envi

# ---------- Step 1: Save ENVI hyperspectral image ----------
def save_envi_with_wavelengths(stacked_array, output_base, wavelengths_nm=None):
    bands, height, width = stacked_array.shape
    if wavelengths_nm is None:
        wavelengths_nm = np.linspace(2100, 2400, bands)

    profile = {
        "driver": "ENVI",
        "dtype": stacked_array.dtype,
        "count": bands,
        "height": height,
        "width": width,
        "crs": None,
    }

    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    img_path = output_base + ".img"
    hdr_path = output_base + ".hdr"

    # Save 3D array of stacked bands into ENVI format .img file
    with rasterio.open(img_path, "w", **profile) as dst:
        for i in range(bands):
            dst.write(stacked_array[i], i + 1)

    # Write wavelengths to .hdr manually
    with open(hdr_path, 'a') as hdr_file:
        hdr_file.write("wavelength = {\n")
        for i, wl in enumerate(wavelengths_nm):
            sep = "," if i < bands - 1 else ""
            hdr_file.write(f"  {wl:.6f}{sep}\n")
        hdr_file.write("}\n")

    return img_path, hdr_path


# ---------- Step 2: Extract wavelengths from saved .hdr file ----------
def read_wavelengths_from_hdr(hdr_path):
    wavelengths = []
    with open(hdr_path, 'r') as f:
        inside = False
        for line in f:
            if "wavelength" in line.lower():
                inside = True
                continue
            if inside:
                if "}" in line:
                    break
                cleaned = line.strip().strip(',')
                try:
                    wavelengths.append(float(cleaned))
                except ValueError:
                    continue
    return np.array(wavelengths)


# ---------- Step 3: Write methane_template.txt ----------
def save_methane_template_from_hdr(hdr_path, output_txt):
    wavelengths = read_wavelengths_from_hdr(hdr_path)
    if len(wavelengths) == 0:
        raise ValueError(f"No wavelengths found in HDR: {hdr_path}")

    center = 2310
    profile = 1.0 - 0.8 * np.exp(-0.5 * ((wavelengths - center) / 20) ** 2)
    
    with open(output_txt, 'w') as f:
        for i, (wl, val) in enumerate(zip(wavelengths, profile)):
            line = f"{wl:.6f} {val:.6f}"
            if i < len(wavelengths) - 1:
                f.write(line + "\n")
            else:
                f.write(line)  # no newline on last line



# ---------- Step 4: Call MAG1C using CLI-style args ----------
def run_mag1c(input_path_base, template_path, output_base, use_gpu=False, iterations=25, group_size=5):
    original_envi_open = spectral.io.envi.open

    # default envi.open() leaves "wavelength" as a list of strings but MAG1C expects a NumPy array of floats
    # this function will fix the wavelength field and return the corrected img
    def patched_envi_open(path, *args, **kwargs):
        img = original_envi_open(path, *args, **kwargs)
        wl = img.metadata.get("wavelength", [])
        if isinstance(wl, list):
            try:
                img.metadata["wavelength"] = np.array([float(w) for w in wl], dtype=np.float64)
            except Exception as e:
                raise ValueError(f"Failed to convert wavelength list to float array: {e}")
        return img

    spectral.io.envi.open = patched_envi_open # overwriting original function

    # Save copy of original CLI args
    original_argv = sys.argv.copy()

    # Use absolute paths
    input_path_base = os.path.abspath(input_path_base)
    template_path = os.path.abspath(template_path)
    output_base = os.path.abspath(output_base)

    sys.argv = [
        "mag1c",
        input_path_base,
        "--spec", template_path,
        "--out", output_base,
        "--iter", str(iterations),
        "--group", str(group_size),
    ]
    if use_gpu:
        sys.argv.append("--gpu")

    # Run MAG1C
    mag1c_module = import_module("mag1c.mag1c")
    mag1c_module.main()

    # Restore CLI args and .open
    sys.argv = original_argv
    spectral.io.envi.open = original_envi_open 

    return output_base + ".img"


# ---------- Step 5: Full pipeline for one folder ----------
def process_one_subdir(sub_dir, output_root, template_path_base):
    # Sort and stack grayscale bands
    file_names = sorted([f for f in os.listdir(sub_dir) if f.endswith(".tif")])
    band_stack = []
    for fname in file_names:
        with rasterio.open(os.path.join(sub_dir, fname)) as src:
            band_stack.append(src.read(1))
    stacked = np.stack(band_stack, axis=0)

    # Make output name
    folder_name = os.path.basename(sub_dir.rstrip(os.sep))
    output_base = os.path.join(output_root, f"mag1c_{folder_name}")
    os.makedirs(output_root, exist_ok=True)

    # Save ENVI image
    img_path, hdr_path = save_envi_with_wavelengths(stacked, output_base)

    # Create matching methane_template.txt
    template_path = os.path.abspath(template_path_base)

    wls = read_wavelengths_from_hdr(hdr_path)
    print("Wavelengths from .hdr:", len(wls))

    save_methane_template_from_hdr(hdr_path, template_path)

    # Run MAG1C
    print(f"Running mag1c on: {sub_dir}")
    output_img = run_mag1c(output_base, template_path, output_base)
    print("MAG1C output saved at:", output_img)
    return output_img


if __name__ == '__main__':
    base_dir = './data/raw_data/STARCOP_train_easy'
    output_root = 'src/mag1c_outputs'
    template_path = 'src/methane_template.txt'

    generator = data_generator(base_dir)

    for i, (images, labels, sub_dir) in enumerate(generator):
        if i >= 1: # change value later
            break
        process_one_subdir(sub_dir, output_root, template_path)
