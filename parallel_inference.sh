#!/bin/bash

# --- Configuration ---
# !!! IMPORTANT: Please verify these paths are correct for your environment !!!

# Base directory for all your input data
MAIN_INPUT_DIR="/glade/campaign/univ/uesf0003/enis/server/nna/real"

# Base directory where processed output will be stored, maintaining sub-structure
MAIN_OUTPUT_DIR="/glade/work/ecoban/uesf0003/real/31m2plxv_new"

# --- Python Script and Model Configuration ---
EDANSA_PROJECT_ROOT="." # Current directory

PYTHON_SCRIPT_REL_PATH="runs/augment/inference.py"
MODEL_REL_PATH="assets/31m2plxv-V1/model_info/best_model_370_val_f1_min=0.8028.pt"
CONFIG_FILE_REL_PATH="assets/31m2plxv-V1/model_info/model_config.json"

# --- Parallelism and Batch Size ---
GPU_COUNT=4
NUM_PARALLEL_JOBS=$((GPU_COUNT * 8))
INFERENCE_BATCH_SIZE=32
echo "NUM_PARALLEL_JOBS: $NUM_PARALLEL_JOBS"
echo "GPU_COUNT: $GPU_COUNT"
echo "INFERENCE_BATCH_SIZE: $INFERENCE_BATCH_SIZE"
# --- Construct full paths ---
PYTHON_SCRIPT_PATH="$EDANSA_PROJECT_ROOT/$PYTHON_SCRIPT_REL_PATH"
MODEL_PATH="$EDANSA_PROJECT_ROOT/$MODEL_REL_PATH"
CONFIG_FILE="$EDANSA_PROJECT_ROOT/$CONFIG_FILE_REL_PATH"

# --- Micromamba Executable Path (Update if necessary) ---
# Attempt to find micromamba in common locations or user's sbin, then PATH
if [ -x "/glade/u/home/ecoban/sbin/micromamba" ]; then
    _MICROMAMBA_EXEC_PATH='/glade/u/home/ecoban/sbin/micromamba'
elif command -v micromamba &> /dev/null; then
    _MICROMAMBA_EXEC_PATH=$(command -v micromamba)
else
    echo "Error: micromamba executable not found in /glade/u/home/ecoban/sbin/micromamba or in PATH." >&2
    echo "Please set _MICROMAMBA_EXEC_PATH manually in the script or ensure micromamba is in your PATH." >&2
    exit 1
fi
echo "Using micromamba executable: $_MICROMAMBA_EXEC_PATH"


# --- Sanity Checks ---
if ! command -v parallel &> /dev/null; then
    echo "Error: GNU Parallel could not be found. Please install it." >&2
    echo "On many Linux systems: sudo apt install parallel  OR  conda install -c conda-forge parallel" >&2
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "Error: Python script not found at '$PYTHON_SCRIPT_PATH'" >&2
    echo "Please ensure this script (parallel_inference.sh) is in your EDANSA project root and paths are correct." >&2
    exit 1
fi
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at '$MODEL_PATH'" >&2
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at '$CONFIG_FILE'" >&2
    exit 1
fi
if [ ! -d "$MAIN_INPUT_DIR" ]; then
    echo "Error: Main input directory not found at '$MAIN_INPUT_DIR'" >&2
    exit 1
fi
# Output directory will be created by each job if it doesn't exist.
# We can make the main output directory here for clarity, though mkdir -p in jobs handles subdirs.
mkdir -p "$MAIN_OUTPUT_DIR" || { echo "Error: Could not create main output directory '$MAIN_OUTPUT_DIR'" >&2; exit 1; }


# --- Activate Micromamba Environment ---
echo "Initializing and activating micromamba environment 'edansa'..."

export MAMBA_ROOT_PREFIX="/glade/work/ecoban/micromamba"
# Define a local __mamba_exe function that points to the absolute path
__mamba_exe() {
    "$_MICROMAMBA_EXEC_PATH" "${@}"
}

# Source the micromamba shell hook
# The eval command is crucial for correctly setting up the shell environment
eval "$(__mamba_exe shell hook --shell bash)"
if [ $? -ne 0 ]; then
    echo "Error: Failed to evaluate micromamba shell hook." >&2
    exit 1
fi

# Activate the desired environment
micromamba activate edansa
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate micromamba environment 'edansa'." >&2
    exit 1
fi
echo "Micromamba environment 'edansa' activated."
echo "Current Python: $(command -v python)"
echo "Current Conda Env: $CONDA_DEFAULT_ENV"


# --- Function to generate folder pairs ---
# This function finds directories named like "20xx" under the MAIN_INPUT_DIR
# and generates the corresponding input and output paths.
generate_folder_pairs() {
    local base_dir="$MAIN_INPUT_DIR"
    local output_base_dir="$MAIN_OUTPUT_DIR"
    # List the specific folders you want to process
    # "dalton" "prudhoe"  "dempster" "ivvavik" "anwr"
    local folders_to_process=("prudhoe")
    for folder in "${folders_to_process[@]}"; do
        local current_input_base="$base_dir/$folder"
        if [ -d "$current_input_base" ]; then
            # Find 20xx directories within the current folder
            find "$current_input_base" -type d -name "20[0-9][0-9]" | while read -r input_leaf_dir; do
                # Calculate relative path from the *main* input dir
                # This ensures the output structure includes the folder name (e.g., anwr/2010)
                local relative_path="${input_leaf_dir#$base_dir}"
                relative_path="${relative_path#/}" # Remove leading slash if present
                local output_leaf_dir="$output_base_dir/$relative_path"
                echo "$input_leaf_dir $output_leaf_dir"
            done
        else
            echo "Warning: Input directory $current_input_base not found, skipping." >&2
        fi
    done
}

# --- Define the Inference Job Function ---
# MODIFIED: Added slot_num parameter and GPU assignment logic
run_inference_job() {
    local input_folder="$1"
    local output_folder="$2"
    local job_num="$3" # GNU Parallel's job sequence number
    local slot_num="$4" # GNU Parallel's slot number {%}

    # Calculate GPU ID: 0 for slots 1-5, 1 for slots 6-10
    local gpu_id=$(( (slot_num - 1) % GPU_COUNT ))

    echo "[Job ${job_num} Slot ${slot_num} GPU ${gpu_id} $(date +'%Y-%m-%d %H:%M:%S')] Starting: Input='${input_folder}', Output='${output_folder}'"
    mkdir -p "${output_folder}"
    if [ $? -ne 0 ]; then
        echo "[Job ${job_num} GPU ${gpu_id} $(date +'%Y-%m-%d %H:%M:%S')] Error: Failed to create output directory '${output_folder}'" >&2
        return 1 # Indicate failure to parallel
    fi

    # MODIFIED: Set CUDA_VISIBLE_DEVICES before calling python
    # Ensure python from the activated env is used
    CUDA_VISIBLE_DEVICES=$gpu_id python "$PYTHON_SCRIPT_PATH" \
        --model_path "$MODEL_PATH" \
        --config_file "$CONFIG_FILE" \
        --input_folder "${input_folder}" \
        --output_folder "${output_folder}" \
        --inference_batch_size "$INFERENCE_BATCH_SIZE" \
        --log-level WARNING

    local py_exit_code=$?
    if [ $py_exit_code -ne 0 ]; then
        echo "[Job ${job_num} GPU ${gpu_id} $(date +'%Y-%m-%d %H:%M:%S')] Error: Python script failed with exit code $py_exit_code for Input='${input_folder}'" >&2
    else
        echo "[Job ${job_num} GPU ${gpu_id} $(date +'%Y-%m-%d %H:%M:%S')] Finished: Input='${input_folder}' (Exit Code: $py_exit_code)"
    fi
    return $py_exit_code # Return python script's exit code
}

# Export the function so GNU Parallel can use it in subshells
export -f run_inference_job
export _MICROMAMBA_EXEC_PATH # Export this so the function can find it if needed in subshells
export PYTHON_SCRIPT_PATH MODEL_PATH CONFIG_FILE INFERENCE_BATCH_SIZE # Export needed vars
export GPU_COUNT
# --- Main execution ---
echo ""
echo "Starting parallel inference processing..."
echo "Number of parallel jobs: $NUM_PARALLEL_JOBS"
echo "Input base directory:    $MAIN_INPUT_DIR"
echo "Output base directory:   $MAIN_OUTPUT_DIR"
echo "Python script:         $PYTHON_SCRIPT_PATH"
echo "Model path:            $MODEL_PATH"
echo "Config file:           $CONFIG_FILE"
echo ""

# Generate folder pairs and pipe them to GNU Parallel
# --env _ ensures the current environment (including activated micromamba and exported function) is passed.
# {#} is the job sequence number
# {%} is the slot number (1 to N)
# {1} is the first field from input (input_folder)
# {2} is the second field from input (output_folder)
# --halt now,fail=1 : Stop all jobs if one fails
# MODIFIED: Changed -j value and added {%} to the command
generate_folder_pairs | parallel --colsep ' ' \
    -j "$NUM_PARALLEL_JOBS" \
    --halt now,fail=1 \
    --env _ \
    run_inference_job {1} {2} {#} {%} # Pass input_folder, output_folder, job_num, and slot_num

PARALLEL_EXIT_CODE=$?

echo ""
if [ $PARALLEL_EXIT_CODE -eq 0 ]; then
    echo "All processing tasks completed successfully by GNU Parallel."
else
    echo "GNU Parallel finished with errors (Exit Code: $PARALLEL_EXIT_CODE). Some jobs may have failed." >&2
    echo "Check the output above for specific error messages from the jobs." >&2
fi
echo "Monitor the output above for progress and any errors."

exit $PARALLEL_EXIT_CODE