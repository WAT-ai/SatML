import os
import numpy as np
import rasterio
import spectral.io.envi
from importlib import import_module
import sys
from image_utils import data_generator

def extract_wavelength_from_filename(filename):
    """
    Extracts 1 wavelength (in nm) from filenames like:
    TOA_AVIRIS_2310nm.tif
    TOA_WV3_SWIR2.tif
    """
    basename = os.path.basename(filename)
    if "nm" in basename:
        try:
            # first part splits filename into parts stored in a list
            # second part just gets the last part of list which has the number "2310nm.tif"
            # last part removes "nm.tif"
            return float(basename.split("_")[-1].replace("nm.tif", ""))
        except ValueError:
            return None
    
    if 'SWIR' in basename:
        swir_map = {
            "SWIR1": 1210,
            "SWIR2": 1570,
            "SWIR3": 1660,
            "SWIR4": 1730,
            "SWIR5": 2165,
            "SWIR6": 2205,
            "SWIR7": 2260,
            "SWIR8": 2330,
        }
        for key, wl in swir_map.items():
            if key in basename:
                return wl
            
    return None


def save_envi_with_wavelengths(stacked_array, output_path, wavelengths_nm):
    bands, height, width = stacked_array.shape

    profile = {
        "driver": "ENVI",
        "dtype": stacked_array.dtype,
        "count": bands,
        "height": height,
        "width": width,
        "crs": None,
    }

    img_path = output_path + ".img"
    hdr_path = output_path + ".hdr"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with rasterio.open(img_path, "w", **profile) as dst:
        for i in range(bands):
            dst.write(stacked_array[i], i + 1)

    with open(hdr_path, 'a') as hdr_file:
        hdr_file.write("wavelength = {\n") # tells ENVI where wavelengths start
        for i, wl in enumerate(wavelengths_nm): # writes out each wavelength on a new line and a comma follows after the wavelength
            sep = "," if i < len(wavelengths_nm) - 1 else ""
            hdr_file.write(f"  {wl:.6f}{sep}\n")
        hdr_file.write("}\n") # closes the wavelength = {...} block

    return img_path, hdr_path


def save_synthetic_methane_template(filepath, wavelengths_nm):
    center = 2310
    profile = 1.0 - 0.8 * np.exp(-0.5 * ((np.array(wavelengths_nm) - center) / 20) ** 2) # Gaussian distribution of value of absorption of methane gas for specifc wavelength

    sorted_pairs = sorted(zip(wavelengths_nm, profile)) # combines 2 lists into a list of sorted pairs based on the first, wavelengths value [(wavelengths, profile), ...]
    wavelengths_sorted, profile_sorted = zip(*sorted_pairs) # unzips the sorted list back into 2 lists

    with open(filepath, 'w') as f:
        for i, (wl, val) in enumerate(zip(wavelengths_sorted, profile_sorted)):
            newline = "\n" if i < len(wavelengths_nm) - 1 else ""
            f.write(f"{wl:.6f} {val:.6f}{newline}")

    return filepath


def run_mag1c(input_path_base, template_path, output_base, use_gpu=False, iterations=25, group_size=5):
    original_envi_open = spectral.io.envi.open

    def patched_envi_open(path, *args, **kwargs):
        img = original_envi_open(path, *args, **kwargs)
        wl = img.metadata.get("wavelength", [])
        if isinstance(wl, list):
            try:
                img.metadata["wavelength"] = np.array([float(w) for w in wl], dtype=np.float64)
            except Exception as e:
                raise ValueError(f"Failed to convert wavelength list to float array: {e}")
        return img

    spectral.io.envi.open = patched_envi_open

    original_argv = sys.argv.copy()
    input_path_base = os.path.abspath(input_path_base)
    template_path = os.path.abspath(template_path)
    output_base = os.path.abspath(output_base)

    sys.argv = [ # simulates CLI arguments
        "mag1c",
        input_path_base,
        "--spec", template_path,
        "--out", output_base,
        "--iter", str(iterations),
        "--group", str(group_size),
    ]
    if use_gpu:
        sys.argv.append("--gpu")

    mag1c_module = import_module("mag1c.mag1c")
    mag1c_module.main()

    sys.argv = original_argv
    spectral.io.envi.open = original_envi_open

    return output_base + ".img"


def process_one_subdir(sub_dir, output_root, template_path):
    ignore_files = {"label_rgba.tif", "labelbinary.tif", "mag1c.tif", "weight_mag1c.tif"}
    file_names = sorted([ 
        f for f in os.listdir(sub_dir) # keeps the file iff
        if f.endswith(".tif") and f not in ignore_files # the file ends with .tif and is not in the ignore_files set
    ])

    bands = [] # hyperspectral cube
    wavelengths = [] 
    for fname in file_names: # loop through all the files in the folder to extract all 16 wavelengths of the image and store it into 
        wl = extract_wavelength_from_filename(fname) # returns a single wl value
        if wl is not None: 
            with rasterio.open(os.path.join(sub_dir, fname)) as src: 
                bands.append(src.read(1)) #src.read(1) reads the first band from the grayscale .tif file as a 2D (512 x 512) array
            wavelengths.append(wl) # save the single wl value to the wavelengths list

    stacked = np.stack(bands, axis=0) # stacks all the 2D NumPy arrays into a 3D hyperspectral cube with shape (num_bands, height, width). stacks 2D images vertically along axis 0 for 1st dimension is num of bands

    folder_name = os.path.basename(sub_dir.rstrip(os.sep)) # removes trailing slash or backslash from folder path
    output_base = os.path.join(output_root, f"mag1c_{folder_name}") # extracts last part of the path, e.g. mag1c_ang20190922t192642_r10240_c0_w512_g512
    save_envi_with_wavelengths(stacked, output_base, wavelengths)
    save_synthetic_methane_template(template_path, wavelengths)
    return run_mag1c(output_base, template_path, output_base)


if __name__ == '__main__':
    base_dir = './data/raw_data/STARCOP_train_easy'
    template_path = "src/methane_template.txt"
    output_root = "src/mag1c_outputs"
    os.makedirs(output_root, exist_ok=True)

    gen = data_generator(base_dir) # stores full dataset into variable gen
    for i, (_, _, sub_dir) in enumerate(gen):
        if i >= 1:
            break
        print("Running mag1c on:", sub_dir)
        result = process_one_subdir(sub_dir, output_root, template_path)
        print("MAG1C output saved at:", result)
