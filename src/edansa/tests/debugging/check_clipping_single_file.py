import numpy as np
import sys
import os
import warnings
import torch # Make sure torch is imported
import csv # Import CSV module
import traceback # For detailed error logging
import glob # Ensure glob is imported for the main section

try:
    from edansa.dataimport import Audio
    from edansa.clippingutils import get_clipping_percent_file as new_get_clipping_percent_file
    from edansa.clippingutils import get_clipping_percent as original_get_clipping_percent
    # import edansa.audio # For edansa.audio.load used by Audio.load_data -> This seems to be implicitly handled by dataimport
except ImportError as e:
    print(f"Error importing EDANSA modules: {e}")
    print(f"This might be due to incorrect path configuration or missing EDANSA installation.")
    sys.exit(1)

def analyze_file_properties(filepath_to_analyze):
    """
    Analyzes a single audio file and returns a dictionary of its properties.
    """
    results = {
        'filepath': filepath_to_analyze,
        'load_status': 'Failure', 'load_error': '',
        'audio_shape': '', 'sample_rate': '', 'audio_dtype': '', 
        'max_abs_val_norm': '', 'norm_note': '',
        'new_segment_len_s': '', 'new_clip_thresh': '', 
        'new_max_segment_clip_pct': '', 'new_method_warnings': '', 'new_method_error': '',
        'orig_clip_thresh': '', 'orig_overall_clip_pct_ch1': '', 
        'orig_overall_clip_pct_ch2': '', 'orig_method_warnings': '', 'orig_method_error': ''
    }

    if not os.path.exists(filepath_to_analyze):
        results['load_error'] = "File not found"
        return results

    # --- 1. Load audio ---
    loaded_audio_data = None
    loaded_sr = None
    try:
        audio_obj = Audio(path=filepath_to_analyze)
        loaded_audio_data, loaded_sr = audio_obj.load_data(
            dtype=torch.float32,
            mono=False, # Assuming stereo based on original script context
            normalize=True,
            channels_first=True
        )

        if loaded_audio_data is None or loaded_sr is None:
            results['load_error'] = "Audio.load_data returned None for audio or sample rate"
            return results # Cannot proceed if basic loading fails

        results['load_status'] = 'Success'
        results['audio_shape'] = str(loaded_audio_data.shape)
        results['sample_rate'] = loaded_sr
        results['audio_dtype'] = str(loaded_audio_data.dtype)
        
        if loaded_audio_data.ndim > 0 and loaded_audio_data.numel() > 0:
            max_abs_val = torch.max(torch.abs(loaded_audio_data)).item()
            results['max_abs_val_norm'] = f"{max_abs_val:.4f}"
            if max_abs_val > 1.0 + 1e-5: # Allow tiny floating point inaccuracies
                 results['norm_note'] = "Max abs value > 1.0. Normalization might not have scaled to [-1,1]."
            else:
                 results['norm_note'] = "Max abs value <= 1.0. Normalization appears to have worked as expected."
        else:
            # This case should ideally not happen if load_data succeeded and returned tensors
            results['load_error'] = "Loaded audio data is empty or has no dimensions, despite successful load call."
            results['load_status'] = 'Partial Failure' # Or keep as success if shape/sr are valid but data is empty
            return results # Cannot proceed with analysis if data is effectively unusable

    except Exception as e:
        results['load_error'] = f"Audio loading exception: {str(e)}\n{traceback.format_exc()}"
        return results # Cannot proceed without loaded data

    # --- 2. Test with NEW edansa.clippingutils.get_clipping_percent_file ---
    segment_len_sec = 10.0
    abs_clipping_threshold_new = 0.999
    results['new_segment_len_s'] = segment_len_sec
    results['new_clip_thresh'] = abs_clipping_threshold_new
    try:
        clipping_percentages_new = None # Initialize
        with warnings.catch_warnings(record=True) as w_gcpf_new:
            warnings.simplefilter("always") # Capture all warnings
            clipping_percentages_new = new_get_clipping_percent_file(
                audio_array=loaded_audio_data, 
                sr=loaded_sr,
                segment_len=segment_len_sec,
                clipping_threshold=abs_clipping_threshold_new
            )
            if w_gcpf_new:
                results['new_method_warnings'] = "; ".join([str(warn_item.message) for warn_item in w_gcpf_new])
        
        if clipping_percentages_new is not None and clipping_percentages_new.numel() > 0:
             results['new_max_segment_clip_pct'] = f"{torch.max(clipping_percentages_new).item():.4f}"
        elif clipping_percentages_new is not None: # Empty tensor
            results['new_max_segment_clip_pct'] = "0.0000 (or empty result)" 
        else: # Function returned None
            results['new_max_segment_clip_pct'] = "N/A (None returned)"
            results['new_method_error'] = "Function returned None"

    except Exception as e:
        results['new_method_error'] = f"NEW method exception: {str(e)}\n{traceback.format_exc()}"

    # --- 3. Test with ORIGINAL edansa.clippingutils.get_clipping_percent ---
    abs_clipping_threshold_orig = 0.999 
    results['orig_clip_thresh'] = abs_clipping_threshold_orig
    try:
        clipping_percentage_orig = None # Initialize
        with warnings.catch_warnings(record=True) as w_gcp_orig:
            warnings.simplefilter("always") # Capture all warnings
            clipping_percentage_orig = original_get_clipping_percent(
                sound_array=loaded_audio_data, 
                threshold=abs_clipping_threshold_orig
            )
            if w_gcp_orig:
                results['orig_method_warnings'] = "; ".join([str(warn_item.message) for warn_item in w_gcp_orig])
        
        if clipping_percentage_orig is not None:
            if clipping_percentage_orig.ndim == 1:
                if clipping_percentage_orig.numel() == 2: # Expected stereo from mono=False load
                    results['orig_overall_clip_pct_ch1'] = f"{clipping_percentage_orig[0].item():.6f}" # Increased precision
                    results['orig_overall_clip_pct_ch2'] = f"{clipping_percentage_orig[1].item():.6f}" # Increased precision
                elif clipping_percentage_orig.numel() == 1: # If it somehow loaded as mono
                    results['orig_overall_clip_pct_ch1'] = f"{clipping_percentage_orig[0].item():.6f}"
                    results['orig_overall_clip_pct_ch2'] = "N/A (Mono)"
                else: # Other 1D tensor
                    results['orig_overall_clip_pct_ch1'] = f"Unexpected 1D tensor: {clipping_percentage_orig}"
                    results['orig_overall_clip_pct_ch2'] = ""
            elif clipping_percentage_orig.numel() == 0: # Empty tensor
                 results['orig_overall_clip_pct_ch1'] = "0.000000 (or empty result)"
                 results['orig_overall_clip_pct_ch2'] = "" 
            else: # Unexpected shape (e.g. >1D)
                results['orig_overall_clip_pct_ch1'] = f"Unexpected shape: {clipping_percentage_orig.shape}"
                results['orig_overall_clip_pct_ch2'] = ""
        else: # Function returned None
            results['orig_overall_clip_pct_ch1'] = "N/A (None returned)"
            results['orig_overall_clip_pct_ch2'] = "N/A (None returned)"
            results['orig_method_error'] = "Function returned None" if not results['orig_method_error'] else results['orig_method_error']

    except Exception as e:
        results['orig_method_error'] = f"ORIGINAL method exception: {str(e)}\n{traceback.format_exc()}"

    return results

if __name__ == "__main__":
    # Define the glob pattern for input files
    # Original path from user: /glade/campaign/univ/uesf0003/enis/server/nna/real/prudhoe/11/2022/*.flac
    # Adjust this path if your files are elsewhere or provide it as a command-line argument for flexibility.
    input_file_pattern = "/glade/campaign/univ/uesf0003/enis/server/nna/real/prudhoe/11/2022/*.flac"
    # Example for local testing (create this directory and put some test FLAC files):
    # input_file_pattern = "./test_audio_data/*.flac" 
    # if not os.path.exists("./test_audio_data"):
    #     os.makedirs("./test_audio_data", exist_ok=True)
    #     print("Created ./test_audio_data for local testing. Please add FLAC files there or change pattern.")

    output_csv_file = "audio_clipping_analysis.csv"

    csv_headers = [
        'filepath',
        'load_status', 'load_error',
        'audio_shape', 'sample_rate', 'audio_dtype', 
        'max_abs_val_norm', 'norm_note',
        'new_segment_len_s', 'new_clip_thresh', 
        'new_max_segment_clip_pct', 'new_method_warnings', 'new_method_error',
        'orig_clip_thresh', 'orig_overall_clip_pct_ch1', 
        'orig_overall_clip_pct_ch2', 'orig_method_warnings', 'orig_method_error'
    ]

    print(f"Starting audio analysis. Results will be saved to: {os.path.abspath(output_csv_file)}")
    
    file_list = glob.glob(input_file_pattern)
    if not file_list:
        print(f"No audio files found matching the pattern: {input_file_pattern}")
        print("Please check the path and ensure there are .flac files present.")
        sys.exit(0) # Exit gracefully if no files
    
    print(f"Found {len(file_list)} files to analyze.")

    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()

        for i, filepath in enumerate(file_list):
            print(f"Processing file {i+1}/{len(file_list)}: {os.path.basename(filepath)}")
            analysis_data = analyze_file_properties(filepath)
            writer.writerow(analysis_data)
            
            # Optional: Flush buffer periodically if processing a very large number of files
            if (i + 1) % 20 == 0: # Flush every 20 files
                csvfile.flush()
                print(f"   ...flushed CSV buffer at file {i+1}...")

    print(f"Analysis complete. All results saved to {os.path.abspath(output_csv_file)}")