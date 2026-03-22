import datetime
import logging
import os
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from tqdm import tqdm

# CSV export checkpoint variables
processed_count = 0
checkpoint_interval = 5
collected_features = []


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def setup_logging():
    """
    Configures the logging system to output to both console and a uniquely named file.
    """
    # Create a unique log filename based on the current timestamp
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
    log_filename = os.path.join(log_dir, f"logs-{timestamp}.log")

    # Get the root logger and set the base level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File Handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Redirect stdout and stderr
    sys.stdout = StreamToLogger(logging.getLogger("STDOUT"), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger("STDERR"), logging.ERROR)

    logging.info(f"Logging initialized. Log file will be saved to: {log_filename}")


def is_tumor_size_sufficient(tumor_mask, min_voxels=300, min_axis_length=5):
    """
    Check if tumor has enough voxels and extent for reliable feature extraction.

    Args:
        tumor_mask: SimpleITK mask image with label 1 for tumor
        min_voxels: Minimum number of voxels required
        min_axis_length: Minimum length along any axis required

    Returns:
        bool: True if tumor is large enough for feature extraction
    """
    # Check total number of voxels
    stats = sitk.StatisticsImageFilter()
    stats.Execute(tumor_mask)
    num_voxels = int(
        stats.GetSum()
    )  # For binary mask with value 1, sum equals voxel count

    if num_voxels < min_voxels:
        return False

    # Check bounding box dimensions
    try:
        lsif = sitk.LabelShapeStatisticsImageFilter()
        lsif.Execute(tumor_mask)
        bounding_box = lsif.GetBoundingBox(1)  # Tumor is labeled with '1'

        # Get bounding box size
        bb_size = bounding_box[3:]  # Extract size (x, y, z)

        # Check if any dimension is too small
        if any(s < min_axis_length for s in bb_size):
            return False

        return True
    except:
        return False  # If we can't calculate bbox, assume it's too small


def load_and_process_images(ct_path, mask_path):
    """
    Loads CT and mask images and processes the mask to:
    1. Keep only tumor label (2)
    2. Separate multiple tumor regions using connected components

    Args:
        ct_path (str): The file path to the CT scan
        mask_path (str): The file path to the segmentation mask

    Returns:
        tuple: (CT image, list of tumor masks)
    """
    try:
        # Load CT image as float32 for feature extraction
        ct_image = sitk.ReadImage(ct_path, sitk.sitkFloat32)

        # Load mask image
        mask_image = sitk.ReadImage(mask_path, sitk.sitkUInt8)

        # Create binary image containing only tumor (label 2)
        tumor_mask = sitk.BinaryThreshold(
            mask_image,
            lowerThreshold=2,
            upperThreshold=2,
            insideValue=1,
            outsideValue=0,
        )

        # Check if there's any tumor present
        stats_filter = sitk.StatisticsImageFilter()
        stats_filter.Execute(tumor_mask)
        if stats_filter.GetSum() == 0:
            print(f"No tumor found in mask: {mask_path}")
            return ct_image, []

        # Use connected component analysis to separate individual tumors
        connected_comp_filter = sitk.ConnectedComponentImageFilter()
        connected_comp_filter.SetFullyConnected(False)  # 26-connectivity in 3D
        connected_tumors = connected_comp_filter.Execute(tumor_mask)

        # Get number of tumor regions
        num_tumors = connected_comp_filter.GetObjectCount()

        if num_tumors == 0:
            return ct_image, []

        # Create a list to hold separate tumor masks
        tumor_masks = []

        # Extract each tumor as a separate mask
        for tumor_label in range(1, num_tumors + 1):
            single_tumor = sitk.BinaryThreshold(
                connected_tumors,
                lowerThreshold=tumor_label,
                upperThreshold=tumor_label,
                insideValue=1,
                outsideValue=0,
            )

            # Only add tumors that are large enough
            if is_tumor_size_sufficient(single_tumor):
                tumor_masks.append(single_tumor)
            else:
                # Count but don't add small tumors
                stats = sitk.StatisticsImageFilter()
                stats.Execute(single_tumor)
                num_voxels = int(stats.GetSum())
                print(f"Skipping small tumor with {num_voxels} voxels in {mask_path}")

        return ct_image, tumor_masks

    except Exception as e:
        print(
            f"Error loading or processing images for CT: {ct_path}, Mask: {mask_path}. Error: {e}"
        )
        return None, []


def partition_and_extract_features(
    extractor, ct_image, tumor_mask, patient_id, mask_id, tumor_id, grid_size=(3, 3, 3)
):
    """
    Partitions a tumor into a grid, and extracts radiomic features.

    Args:
        extractor: The pre-initialized feature extractor object
        ct_image: The CT image
        tumor_mask: A single tumor mask
        patient_id: The patient identifier
        mask_id: The mask identifier
        tumor_id: The tumor identifier (for multiple tumors)
        grid_size: The dimensions of the grid to partition the tumor into

    Returns:
        tuple: (feature_volume, validity_mask, feature_names)
    """
    try:
        # Get the bounding box of the tumor mask
        lsif = sitk.LabelShapeStatisticsImageFilter()
        lsif.Execute(tumor_mask)
        # Tumor is labeled with '1' in the processed mask
        bounding_box = lsif.GetBoundingBox(1)
    except RuntimeError:
        # This occurs if the mask is empty (no label '1' found)
        tqdm.write(
            f"Warning: Bounding box could not be calculated for Patient {patient_id} ({mask_id}), Tumor {tumor_id}. The mask may be empty."
        )
        return None, None, None

    # Bounding box format: (start_x, start_y, start_z, size_x, size_y, size_z) in voxel indices
    bb_origin_voxel = bounding_box[:3]
    bb_size_voxel = bounding_box[3:]

    # Check if the tumor is too small for partitioning
    if any(s < grid_size[i] for i, s in enumerate(bb_size_voxel)):
        tqdm.write(
            f"Warning: Tumor bounding box {bb_size_voxel} for Patient {patient_id} ({mask_id}), Tumor {tumor_id} is smaller than grid size {grid_size}. Cannot partition."
        )
        return None, None, None

    # Prepare for feature extraction loop
    feature_volume = None
    feature_names = None
    validity_mask = np.zeros(grid_size, dtype=bool)

    # Loop through the grid
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            for k in range(grid_size[2]):
                # Calculate the origin and size of the current cube
                cube_size = [
                    dim // g_dim for dim, g_dim in zip(bb_size_voxel, grid_size)
                ]

                # Distribute remainder voxels among the first cubes
                rem_size = [dim % g_dim for dim, g_dim in zip(bb_size_voxel, grid_size)]

                current_cube_size = list(cube_size)
                current_cube_size[0] += 1 if i < rem_size[0] else 0
                current_cube_size[1] += 1 if j < rem_size[1] else 0
                current_cube_size[2] += 1 if k < rem_size[2] else 0

                offset_i = i * cube_size[0] + min(i, rem_size[0])
                offset_j = j * cube_size[1] + min(j, rem_size[1])
                offset_k = k * cube_size[2] + min(k, rem_size[2])

                crop_origin = (
                    bb_origin_voxel[0] + offset_i,
                    bb_origin_voxel[1] + offset_j,
                    bb_origin_voxel[2] + offset_k,
                )

                # Crop the mask and check if it's empty
                cropped_mask = sitk.RegionOfInterest(
                    tumor_mask, size=current_cube_size, index=crop_origin
                )
                stats = sitk.StatisticsImageFilter()
                stats.Execute(cropped_mask)

                if stats.GetSum() == 0:
                    # This cube has no tumor, mark as invalid and continue
                    validity_mask[i, j, k] = False
                    continue

                # Check if subregion is too small (to avoid segmentation fault)
                if not is_tumor_size_sufficient(
                    cropped_mask, min_voxels=50, min_axis_length=3
                ):
                    validity_mask[i, j, k] = False
                    continue

                # This cube is valid
                validity_mask[i, j, k] = True
                cropped_ct = sitk.RegionOfInterest(
                    ct_image, size=current_cube_size, index=crop_origin
                )

                try:
                    # Execute pyradiomics extractor with error handling
                    result = extractor.execute(cropped_ct, cropped_mask, label=1)

                    # On the first successful extraction, initialize the feature volume
                    if feature_volume is None:
                        feature_names = sorted(
                            [key for key in result if not key.startswith("diagnostics")]
                        )
                        num_features = len(feature_names)
                        feature_volume = np.zeros(
                            (*grid_size, num_features), dtype=np.float32
                        )

                    # Store the feature vector, using .get() for safety against missing keys
                    feature_vector = np.array(
                        [float(result.get(key, np.nan)) for key in feature_names]
                    )
                    feature_volume[i, j, k, :] = feature_vector

                except Exception as e:
                    # Pyradiomics can fail if ROI is too small or has other issues
                    validity_mask[i, j, k] = False
                    tqdm.write(
                        f"Pyradiomics failed on a sub-volume for patient {patient_id} ({mask_id}), Tumor {tumor_id}. Marking as invalid. Error: {e}"
                    )

    # Check if we have at least one valid subvolume
    if np.any(validity_mask):
        return feature_volume, validity_mask, feature_names
    else:
        return None, None, None


def collect_feature_data(
    patient_id, mask_id, tumor_id, feature_volume, validity_mask, feature_names
):
    """
    Collect feature data for CSV export.

    Args:
        patient_id: Patient identifier
        mask_id: Mask identifier
        tumor_id: Tumor identifier
        feature_volume: Feature volume array
        validity_mask: Validity mask array
        feature_names: List of radiomic feature names.

    Returns:
        list: List of feature dictionaries
    """
    feature_records = []

    if feature_volume is None or validity_mask is None:
        return feature_records

    try:
        # Get valid subvolume indices
        valid_indices = np.where(validity_mask)

        # Extract features from valid subvolumes
        for idx in range(len(valid_indices[0])):
            i, j, k = (
                valid_indices[0][idx],
                valid_indices[1][idx],
                valid_indices[2][idx],
            )

            # Create feature record
            feature_record = {
                "patient_id": patient_id,
                "mask_id": mask_id,
                "tumor_id": tumor_id,  # Add tumor ID to track multiple tumors
                "subvolume_i": i,
                "subvolume_j": j,
                "subvolume_k": k,
                "grid_position": f"{i}_{j}_{k}",
            }

            # Add feature values with proper names
            feature_values = feature_volume[i, j, k, :]
            if feature_names and len(feature_names) == len(feature_values):
                # Use actual feature names
                feature_record.update(zip(feature_names, feature_values))
            else:
                # Fallback to generic names
                for feat_idx, feat_value in enumerate(feature_values):
                    feature_record[f"feature_{feat_idx}"] = feat_value

            feature_records.append(feature_record)

    except Exception as e:
        tqdm.write(
            f"Error collecting features for {patient_id} ({mask_id}), Tumor {tumor_id}: {e}"
        )

    return feature_records


def export_features_to_csv(features_list, output_path):
    """
    Appends collected features to a single CSV file.

    Args:
        features_list (list): List of feature dictionaries to append.
        output_path (str): The path to the single CSV file.
    """
    if not features_list:
        return

    try:
        # Create DataFrame from the new features
        df = pd.DataFrame(features_list)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write header only if the file is new or empty
        header_needed = (
            not os.path.exists(output_path) or os.path.getsize(output_path) == 0
        )
        df.to_csv(output_path, mode="a", header=header_needed, index=False)
        tqdm.write(
            f"Checkpoint: Appended {len(features_list)} feature records to {output_path}"
        )

    except Exception as e:
        tqdm.write(f"Error exporting features to CSV: {e}")


if __name__ == "__main__":
    # Setup logging
    setup_logging()

    # Define paths
    METADATA_PATH = "data/metadata.csv"
    FEATURES_CSV_PATH = "out/features.csv"
    CT_PARAMS_PATH = "config/exampleCT.yaml"

    # Pre-run check for essential files
    if not os.path.exists(METADATA_PATH):
        print(f"FATAL: Metadata file not found at '{METADATA_PATH}'.")
        exit(1)
    if not os.path.exists(CT_PARAMS_PATH):
        print(f"FATAL: PyRadiomics parameters file not found at '{CT_PARAMS_PATH}'.")
        exit(1)

    # Initialize the Feature Extractor
    try:
        print("Initializing PyRadiomics feature extractor from parameters file...")
        feature_extractor = featureextractor.RadiomicsFeatureExtractor(CT_PARAMS_PATH)
        print("Extractor initialized successfully.")
    except Exception as e:
        print(f"FATAL: Could not initialize feature extractor. Error: {e}")
        exit(1)

    # Load the metadata file
    try:
        metadata_df = pd.read_csv(METADATA_PATH)
    except FileNotFoundError:
        print(f"FATAL: Metadata file not found at '{METADATA_PATH}'.")
        print(
            "Please ensure you are running this script from the project's root directory."
        )
        exit(1)

    # Checkpoint: Load already processed IDs
    processed_ids = set()
    if os.path.exists(FEATURES_CSV_PATH):
        try:
            if os.path.getsize(FEATURES_CSV_PATH) > 0:
                processed_df = pd.read_csv(FEATURES_CSV_PATH)
                if all(
                    col in processed_df.columns
                    for col in ["patient_id", "mask_id", "tumor_id"]
                ):
                    processed_ids = set(
                        zip(
                            processed_df["patient_id"].astype(str),
                            processed_df["mask_id"].astype(str),
                            processed_df["tumor_id"].astype(str),
                        )
                    )
                    print(
                        f"Resuming. Found {len(processed_ids)} already processed patient-mask-tumor combinations."
                    )
                else:
                    print(
                        f"Warning: '{FEATURES_CSV_PATH}' is missing required columns. Starting from scratch."
                    )
            else:
                print(
                    f"Warning: '{FEATURES_CSV_PATH}' exists but is empty. Starting from scratch."
                )
        except (pd.errors.ParserError, KeyError) as e:
            print(
                f"Warning: Could not parse '{FEATURES_CSV_PATH}'. It may be corrupted. Error: {e}. Starting from scratch."
            )

    print("Starting tumor feature extraction...")

    # Process each patient
    for index, row in tqdm(
        metadata_df.iterrows(), total=metadata_df.shape[0], desc="Processing Patients"
    ):
        try:
            patient_id = str(row["patient_id"])
            ct_path = str(row["ct_scan_path"])
            mask_path = str(row["mask_path"])

            mask_basename = os.path.basename(mask_path)
            mask_id = mask_basename.split(".")[0]

            # 1. Load and Process Images (get CT and separate tumor masks)
            ct_image, tumor_masks = load_and_process_images(ct_path, mask_path)

            if ct_image is None or not tumor_masks:
                tqdm.write(
                    f"Skipping Patient {patient_id} ({mask_id}): No CT or tumor regions found."
                )
                continue

            tqdm.write(
                f"Found {len(tumor_masks)} tumor regions for Patient {patient_id} ({mask_id})"
            )

            # 2. Process each tumor region separately
            for tumor_idx, tumor_mask in enumerate(tumor_masks):
                try:
                    tumor_id = f"tumor_{tumor_idx + 1}"

                    # Skip if already processed
                    if (patient_id, mask_id, tumor_id) in processed_ids:
                        tqdm.write(
                            f"Skipping (already processed): Patient {patient_id}, Mask {mask_id}, {tumor_id}"
                        )
                        continue

                    # Extract features for this tumor
                    feature_volume, validity_mask, feature_names = (
                        partition_and_extract_features(
                            feature_extractor,
                            ct_image,
                            tumor_mask,
                            patient_id,
                            mask_id,
                            tumor_id,
                            grid_size=(3, 3, 3),
                        )
                    )

                    # Collect and handle features
                    if feature_volume is not None:
                        # Collect features for CSV export
                        feature_records = collect_feature_data(
                            patient_id,
                            mask_id,
                            tumor_id,
                            feature_volume,
                            validity_mask,
                            feature_names,
                        )
                        collected_features.extend(feature_records)

                        # Increment processed count
                        processed_count += 1

                        # Export to CSV every `checkpoint_interval` patients/tumors
                        if (
                            processed_count % checkpoint_interval == 0
                            and collected_features
                        ):
                            export_features_to_csv(
                                collected_features, FEATURES_CSV_PATH
                            )
                            collected_features = []  # Reset for next batch

                        tqdm.write(
                            f"Success: Processed Patient {patient_id} ({mask_id}), {tumor_id}"
                        )
                    else:
                        # This occurs if the tumor was too small or no features could be extracted
                        tqdm.write(
                            f"Warning: Could not extract any valid features for Patient {patient_id} ({mask_id}), {tumor_id}. Skipping."
                        )
                except Exception as e:
                    # Catch exceptions during tumor processing to ensure the script continues
                    tqdm.write(
                        f"Error processing tumor {tumor_idx + 1} for Patient {patient_id} ({mask_id}): {e}"
                    )
                    tqdm.write("Continuing with next tumor...")
        except Exception as e:
            # Catch exceptions during patient processing to ensure the script continues
            tqdm.write(f"Error processing Patient {patient_id} ({mask_id}): {e}")
            tqdm.write("Continuing with next patient...")

    print("\nTumor feature extraction complete.")

    # Export any remaining features
    if collected_features:
        export_features_to_csv(collected_features, FEATURES_CSV_PATH)
        tqdm.write(f"Exported final batch of {len(collected_features)} features.")

    print(f"\nAll tumor features have been saved to '{FEATURES_CSV_PATH}'.")
