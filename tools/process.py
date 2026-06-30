import os
import numpy as np
import nibabel as nib
from scipy.ndimage import distance_transform_edt, label as nd_label
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import time


def find_nearest_label(mask, label, return_indices=False):
    """
    For each voxel in mask that is 1, find the nearest label value.
    
    Args:
        mask: Binary mask array
        label: Label array
        return_indices: If True, return the indices as well
    
    Returns:
        Array with nearest label values for mask positions
    """
    # Get positions where mask is 1
    mask_positions = np.where(mask > 0)
    
    # Get positions where label has values
    label_positions = np.where(label > 0)
    
    if len(label_positions[0]) == 0:
        # No labels to reference
        return np.zeros_like(mask)
    
    # For each mask position, find nearest label
    result = np.zeros_like(mask)
    
    # Create distance transform for each unique label value
    unique_labels = np.unique(label[label > 0])
    
    for mask_idx in range(len(mask_positions[0])):
        i, j, k = mask_positions[0][mask_idx], mask_positions[1][mask_idx], mask_positions[2][mask_idx]
        
        # Find nearest non-zero label
        min_dist = float('inf')
        nearest_label_value = 0
        
        for label_idx in range(len(label_positions[0])):
            li, lj, lk = label_positions[0][label_idx], label_positions[1][label_idx], label_positions[2][label_idx]
            dist = np.sqrt((i - li)**2 + (j - lj)**2 + (k - lk)**2)
            
            if dist < min_dist:
                min_dist = dist
                nearest_label_value = label[li, lj, lk]
        
        result[i, j, k] = nearest_label_value
    
    return result


def remove_noise_by_connectivity(seg, mask, expected_components={1: 2, 2: 1, 3: 2, 4: 2, 5: 1, 6: 2, 7: 1}, verbose=False):
    """
    Remove noise based on connected component analysis (optimized).
    
    For each label value, keep only the N largest connected components,
    where N is specified in expected_components. Noise voxels are handled as:
    - If inside mask: filled with nearest valid label
    - If outside mask: set to 0
    
    Args:
        seg: Segmentation array
        mask: Binary mask array
        expected_components: Dict mapping label value to expected number of components
        verbose: If True, print detailed information
    
    Returns:
        Tuple of (cleaned_seg, noise_inside_mask, noise_outside_mask)
    """
    result = np.copy(seg)
    noise_mask_inside = np.zeros_like(seg, dtype=bool)
    noise_mask_outside = np.zeros_like(seg, dtype=bool)
    
    for label_val, expected_count in expected_components.items():
        # Find all voxels with this label
        label_mask = (seg == label_val)
        
        if not np.any(label_mask):
            continue
        
        # Perform connected component analysis
        labeled_components, num_components = nd_label(label_mask)
        
        if num_components <= expected_count:
            if verbose:
                print(f"    Label {label_val}: {num_components} components (expected {expected_count}) - OK")
            continue
        
        if verbose:
            print(f"    Label {label_val}: {num_components} components (expected {expected_count}) - removing {num_components - expected_count} noise regions")
        
        # Use numpy bincount for faster size calculation
        component_sizes = np.bincount(labeled_components.ravel())
        # component_sizes[0] is background, skip it
        component_sizes[0] = 0
        
        # Get indices of largest components
        largest_indices = np.argsort(component_sizes)[-expected_count:]
        keep_components = set(largest_indices)
        
        # Create mask for components to remove (vectorized operation)
        remove_mask = label_mask & ~np.isin(labeled_components, list(keep_components))
        
        # Split into inside/outside mask
        noise_in_mask = remove_mask & (mask > 0)
        noise_out_mask = remove_mask & (mask == 0)
        
        noise_mask_inside |= noise_in_mask
        noise_mask_outside |= noise_out_mask
        
        # Remove from result
        result[remove_mask] = 0
    
    total_noise = np.sum(noise_mask_inside) + np.sum(noise_mask_outside)
    if verbose and total_noise > 0:
        print(f"  - Found {total_noise} noise voxels ({np.sum(noise_mask_inside)} inside mask, {np.sum(noise_mask_outside)} outside mask)")
    
    # Handle noise outside mask (already set to 0)
    # Handle noise inside mask (fill with nearest label)
    if np.any(noise_mask_inside):
        filled_labels = find_nearest_label_fast(noise_mask_inside.astype(np.int16), result)
        result[noise_mask_inside] = filled_labels[noise_mask_inside]
    
    return result, noise_mask_inside, noise_mask_outside


def find_nearest_label_fast(mask, label):
    """
    Fast version using distance transform for finding nearest labels.
    
    Args:
        mask: Binary mask array where we need to fill values
        label: Label array with reference values
    
    Returns:
        Array with nearest label values for mask positions
    """
    result = np.copy(label)
    
    # Find voxels in mask but not in label (need filling)
    mask_only = (mask > 0) & (label == 0)
    
    if not np.any(mask_only):
        return result
    
    # For each unique label value, compute distance transform
    unique_labels = np.unique(label[label > 0])
    
    if len(unique_labels) == 0:
        return result
    
    # Store distances and labels
    min_distances = np.full(mask.shape, np.inf)
    nearest_labels = np.zeros(mask.shape, dtype=label.dtype)  # Use same dtype as label
    
    for label_val in unique_labels:
        # Create binary mask for this label
        label_mask = (label == label_val)
        
        # Compute distance transform
        distances = distance_transform_edt(~label_mask)
        
        # Update nearest label where this is closer
        closer = distances < min_distances
        min_distances[closer] = distances[closer]
        nearest_labels[closer] = int(label_val)  # Ensure integer assignment
    
    # Fill only the mask_only positions
    result[mask_only] = nearest_labels[mask_only]
    
    return result


def align_seg_to_mask(seg_path, mask_path, output_path, verbose=False):
    """
    Align segmentation to original mask with noise removal:
    
    Processing steps:
    1. Remove noise based on connected components (keep largest N components per label)
       - Noise inside mask: filled with nearest valid label
       - Noise outside mask: set to 0
    2. Remove seg voxels where mask=0 (set to 0)
    3. Fill mask voxels where seg=0 with nearest seg label
    
    Args:
        seg_path: Path to segmentation file
        mask_path: Path to original mask file
        output_path: Path to save aligned segmentation
        verbose: If True, print detailed debug information
    """
    # Load files
    seg_nii = nib.load(seg_path)
    seg = seg_nii.get_fdata().astype(np.int16)
    
    mask_nii = nib.load(mask_path)
    mask = mask_nii.get_fdata().astype(np.int16)
    
    # Ensure binary mask
    mask = (mask > 0).astype(np.int16)
    
    # Create result starting from seg
    result = np.copy(seg)
    
    # Step 1: Remove noise based on connected components
    if verbose:
        print("\n  === Step 1: Noise Removal ===")
    result, noise_inside, noise_outside = remove_noise_by_connectivity(result, mask, verbose=verbose)
    
    # Step 2: Remove seg voxels where mask is 0
    seg_only = (result != 0) & (mask == 0)
    result[seg_only] = 0
    if verbose:
        print(f"\n  === Step 2: Removed {np.sum(seg_only)} voxels outside mask ===")
    
    # Step 3: Fill mask voxels where seg is 0 with nearest label
    mask_only = (mask > 0) & (result == 0)
    
    if np.any(mask_only):
        if verbose:
            print(f"\n  === Step 3: Filling {np.sum(mask_only)} missing voxels ===")
        filled_labels = find_nearest_label_fast(mask, result)
        result[mask_only] = filled_labels[mask_only]
    
    # Save result
    result_nii = nib.Nifti1Image(result.astype(np.int16), seg_nii.affine, seg_nii.header)
    nib.save(result_nii, output_path)
    
    if verbose:
        print(f"  - Final result labels: {np.unique(result)}")
    
    return result


def process_single_file(args):
    """
    Process a single file pair (wrapper for parallel processing).
    
    Args:
        args: Tuple of (seg_file, mask_file, output_file, file_index, total_files)
    
    Returns:
        Tuple of (success, filename, error_message, elapsed_time)
    """
    seg_file, mask_file, output_file, file_idx, total_files = args
    
    try:
        start_time = time.time()
        print(f"[{file_idx}/{total_files}] Processing: {os.path.basename(seg_file)}...")
        align_seg_to_mask(seg_file, mask_file, output_file, verbose=False)  # Disable verbose for cleaner output
        elapsed = time.time() - start_time
        print(f"[{file_idx}/{total_files}] ✓ Completed in {elapsed:.2f}s: {os.path.basename(seg_file)}")
        return (True, os.path.basename(seg_file), None, elapsed)
    except Exception as e:
        elapsed = time.time() - start_time if 'start_time' in locals() else 0
        print(f"[{file_idx}/{total_files}] ✗ Error: {os.path.basename(seg_file)} - {str(e)}")
        return (False, os.path.basename(seg_file), str(e), elapsed)


def find_matching_mask(seg_file, mask_dir):
    """
    Find matching mask file for a segmentation file.
    
    Args:
        seg_file: Path to segmentation file
        mask_dir: Directory containing mask files
    
    Returns:
        Path to matching mask file or None
    """
    mask_file = None
    
    # Try exact name match
    for ext in ['.nii.gz', '.nii']:
        potential_mask = mask_dir / f"{seg_file.stem.replace('.nii', '')}{ext}"
        if potential_mask.exists():
            mask_file = potential_mask
            break
    
    if mask_file is None:
        # Try without _seg suffix or similar
        base_name = seg_file.stem.replace('.nii', '').replace('_seg', '').replace('_pred', '').replace('_inference', '')
        for ext in ['.nii.gz', '.nii']:
            potential_mask = mask_dir / f"{base_name}{ext}"
            if potential_mask.exists():
                mask_file = potential_mask
                break
    
    return mask_file


def process_directory(seg_dir, mask_dir, output_dir, num_workers=None):
    """
    Process all segmentation files in a directory with parallel processing.
    
    Args:
        seg_dir: Directory containing segmentation files
        mask_dir: Directory containing original mask files
        output_dir: Directory to save aligned segmentations
        num_workers: Number of parallel workers (default: CPU count - 1)
    """
    seg_dir = Path(seg_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all segmentation files
    seg_files = sorted(seg_dir.glob("*.nii.gz")) + sorted(seg_dir.glob("*.nii"))
    
    if len(seg_files) == 0:
        print(f"No segmentation files found in {seg_dir}")
        return
    
    print(f"Found {len(seg_files)} segmentation files")
    
    # Find matching pairs
    file_pairs = []
    for idx, seg_file in enumerate(seg_files, 1):
        mask_file = find_matching_mask(seg_file, mask_dir)
        
        if mask_file is None:
            print(f"Warning: No matching mask found for {seg_file.name}, skipping...")
            continue
        
        output_file = output_dir / seg_file.name
        file_pairs.append((str(seg_file), str(mask_file), str(output_file), idx, len(seg_files)))
    
    if len(file_pairs) == 0:
        print("No valid file pairs found to process")
        return
    
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"\nProcessing {len(file_pairs)} files with {num_workers} parallel workers...")
    print("=" * 80)
    
    start_time = time.time()
    
    # Process files in parallel
    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = pool.map(process_single_file, file_pairs)
    else:
        # Sequential processing for debugging
        results = [process_single_file(args) for args in file_pairs]
    
    # Summary
    elapsed_total = time.time() - start_time
    successful = sum(1 for success, _, _, _ in results if success)
    failed = len(results) - successful
    
    # Calculate statistics
    successful_times = [elapsed for success, _, _, elapsed in results if success]
    if successful_times:
        avg_time = sum(successful_times) / len(successful_times)
        min_time = min(successful_times)
        max_time = max(successful_times)
    else:
        avg_time = min_time = max_time = 0
    
    print("\n" + "=" * 80)
    print(f"Processing complete!")
    print(f"Total time: {elapsed_total:.2f}s")
    print(f"Average time per file: {avg_time:.2f}s (min: {min_time:.2f}s, max: {max_time:.2f}s)")
    print(f"Successfully processed: {successful}/{len(results)} files")
    
    if num_workers > 1:
        speedup = (avg_time * len(results)) / elapsed_total if elapsed_total > 0 else 0
        print(f"Parallel speedup: {speedup:.2f}x")
    
    if failed > 0:
        print(f"\nFailed: {failed} files")
        print("\nFailed files:")
        for success, filename, error, _ in results:
            if not success:
                print(f"  - {filename}: {error}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Align segmentation masks to reference mask geometry."
    )
    parser.add_argument(
        "--seg_dir", type=str, required=True,
        help="Directory containing segmentation NIfTI files",
    )
    parser.add_argument(
        "--mask_dir", type=str, required=True,
        help="Directory containing reference mask NIfTI files",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save aligned segmentations",
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Parallel workers (default: CPU count - 1)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Segmentation Mask Alignment Tool with Parallel Processing")
    print("=" * 80)
    print(f"Segmentation directory: {args.seg_dir}")
    print(f"Mask directory: {args.mask_dir}")
    print(f"Output directory: {args.output_dir}")

    total_cpus = cpu_count()
    workers = args.num_workers if args.num_workers is not None else max(1, total_cpus - 1)
    print(f"Available CPUs: {total_cpus}")
    print(f"Workers to use: {workers}")
    print("=" * 80)

    if not os.path.exists(args.seg_dir):
        print(f"Error: Segmentation directory does not exist: {args.seg_dir}")
        exit(1)

    if not os.path.exists(args.mask_dir):
        print(f"Error: Mask directory does not exist: {args.mask_dir}")
        exit(1)

    process_directory(args.seg_dir, args.mask_dir, args.output_dir, num_workers=args.num_workers)

    print("\n" + "=" * 80)
    print("All processing complete!")
    print("=" * 80)

