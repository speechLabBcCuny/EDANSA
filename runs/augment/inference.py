import torch
from torchvision import transforms
from typing import Union, Dict, Tuple, Callable
import numpy as np
import logging
import argparse
from pathlib import Path

from edansa import modelarchs, inference, datasets, wandbutils
from edansa import runutils

import edansa.io

# Default values
DEFAULT_CHANNEL_SELECTION = 'clipping'
DEFAULT_SAMPLING_RATE = 48000  # Example, adjust if needed
DEFAULT_EXCERPT_LENGTH = 10  # Example, adjust if needed
DEFAULT_CLIPPING_THRESHOLD = 1.0


def get_config(args: argparse.Namespace) -> Dict:

    # Attempt to load config, passing None for run_identity
    config, run = wandbutils.get_wandb_config(None,
                                              config_file=args.config_file)
    if run is not None:
        # If loaded from wandb, merge run config into main config dict
        config.update(run.config)
    # else: # REMOVED: No explicit else needed
    # If only loaded from file or defaults, set run_id to None or based on args

    # Determine device based on --gpu argument
    # REMOVED: Simplified device selection logic.
    # The device is now set in the main inference.py script via _setup_device_and_config
    # based on the --device argument or default logic.
    # We just ensure the key exists in the config if needed later, or rely on the main script setting it.
    # config['device'] = torch.device(device_str) # Removed

    # --- Set parameters from args or defaults --- #
    # Use getattr to safely get args, falling back to defaults defined above or in existing config
    config['channel_selection_method'] = getattr(
        args, 'channel_selection_method',
        config.get('channel_selection_method', DEFAULT_CHANNEL_SELECTION))
    config['skip_clipping_info'] = getattr(
        args, 'skip_clipping_info', config.get('skip_clipping_info', False))
    # config['v_str'] = args.model_id # REMOVED: model_id arg doesn't exist
    config['model_path'] = args.model_path
    config['sampling_rate'] = config.get('sampling_rate', DEFAULT_SAMPLING_RATE)
    config['excerpt_length'] = config.get('excerpt_length',
                                          DEFAULT_EXCERPT_LENGTH)
    # Ensure arch settings are present if not loaded from wandb
    if 'arch' not in config:
        config['arch'] = {}
    config['arch']['intermediate_pool_type'] = config['arch'].get(
        'intermediate_pool_type', 'avg')  # Example default
    config['arch']['global_pool_type'] = config['arch'].get(
        'global_pool_type', 'avg')  # Example default
    config['arch']['last_conv_block_out_chan'] = config['arch'].get(
        'last_conv_block_out_chan', 512)
    config['arch']['feature_method'] = config['arch'].get(
        'feature_method', 'logmel')
    # Ensure target_taxo and code2excell_names are handled (might require loading taxonomy file?)
    if 'target_taxo' not in config:
        logging.warning(
            "'target_taxo' not found in config. Model output interpretation might be incorrect."
        )
        config['target_taxo'] = []  # Default to empty list
    if 'code2excell_names' not in config:
        logging.warning(
            "'code2excell_names' not found in config. Output column names might be incorrect."
        )
        config['code2excell_names'] = {}  # Default to empty dict
    if 'max_mel_len' not in config:
        config['max_mel_len'] = 1998  # Example default, adjust

    # --- Hardcoded Fixes (Keep or Remove?) --- #
    # REMOVED: Hardcoded fix based on run_id, as run_id is no longer reliably sourced
    # if config.get('run_id') == '1esfp153': # Check using .get() for safety
    #     config['arch']['last_conv_block_out_chan'] = 512

    # --- Validation --- #
    # Ensure critical paths/IDs are present
    if not config.get('model_path'):  # Check using .get()
        raise ValueError("Model path (--model_path) is required.")
    # REMOVED: Warnings comparing run_id/model_id to model_path
    # if config.get('run_id') and config['run_id'] not in config['model_path']:
    #     logging.warning(
    #         f"Run ID '{config['run_id']}' not found in model path '{config['model_path']}'. Ensure consistency."
    #     )
    # elif args.model_id not in config['model_path']:
    #     logging.warning(
    #         f"Model ID '{args.model_id}' not found in model path '{config['model_path']}'. Ensure consistency."
    #     )

    # Convert list-of-lists to list-of-tuples for loc_per_set AFTER loading/merging
    if 'loc_per_set' in config:
        loc_per_set = config['loc_per_set']
        loc_per_set = {
            set_type: [tuple(m) for m in list_regloc]
            for set_type, list_regloc in loc_per_set.items()
        }
        config['loc_per_set'] = runutils.make_locs_caseinsensitive(loc_per_set)
    else:
        config['loc_per_set'] = {}  # Default if missing

    logging.info(f"Loaded configuration: {config}")
    return config


def get_model(
    config: Dict,
    category_count: int,
    map_location_device: torch.device  # Add map_location device argument
) -> Tuple[torch.nn.Module, Dict]:
    """Initializes the model architecture and loads the weights onto the specified device."""
    intermediate_pool_type = config['arch']['intermediate_pool_type']
    global_pool_type = config['arch']['global_pool_type']
    # REMOVED: hard coded fix based on run_id
    # if config.get('run_id') == '1esfp153':
    #     config['arch']['last_conv_block_out_chan'] = 512
    model = modelarchs.Cnn6(None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            category_count,
                            intermediate_pool_type=intermediate_pool_type,
                            global_pool_type=global_pool_type,
                            last_conv_block_out_chan=config['arch'].get(
                                'last_conv_block_out_chan', 512))

    # model = model.float().to(
    #     config['device'])  # REMOVED: Moving to device is handled after loading in the main script
    logging.info(
        f"Loading model weights from {config['model_path']} onto device {map_location_device}"
    )
    model.load_state_dict(
        torch.load(config['model_path'],
                   map_location=map_location_device))  # Use map_location device
    # Move model and set to eval() happens in the main script's _setup_device_and_config
    # model = model.eval() # REMOVED: Moved this after the .to(device) call in main script

    return model, config


# %%
def prepare_dataloader_from_audio_ins(
    mono_data: Union[np.ndarray, torch.Tensor],  # Changed from audio_ins
    config: Dict
) -> Dict[str, torch.utils.data.DataLoader]:
    """Prepares a DataLoader for inference from pre-processed mono audio data."""
    # Removed audio_ins.load_data(...)
    # Removed audio_ins.select_channel(...)

    sr = config['sampling_rate']  # Get sr from config
    # Get the device determined by the main script's setup
    target_device = config.get(
        'device', torch.device('cpu'))  # Default to cpu if key somehow missing

    # --- Use the provided mono_data and sr (Original Padding Logic) --- #
    # Ensure mono_data is a tensor before padding
    if isinstance(mono_data, np.ndarray):
        mono_data_tensor = torch.from_numpy(mono_data).float()
    else:
        mono_data_tensor = mono_data.float()  # Ensure float

    input_file_data = inference.pad_audio(mono_data_tensor,
                                          config['excerpt_length'],
                                          sr)  # Use passed sr

    # divide to 10 second excerpts
    expected_size = sr * config['excerpt_length']  # Use passed sr
    input_file_data = input_file_data.reshape(-1, expected_size)

    # using wav as input
    to_tensor = modelarchs.WaveToTensor(
        config['max_mel_len'],
        sr,  # Use passed sr
        device=target_device,  # Pass the target device for feature extraction
        feature_method=config['arch'].get('feature_method', 'logmel'))

    transformCompose = transforms.Compose([
        to_tensor,
        #     spec_augmenter,
    ])

    batch_transforms = [
        # 'random_mergev2'
    ]

    dataset = {}
    dataset['predict'] = datasets.AugmentingAudioDataset(
        input_file_data,  # Pass the tensor
        None,
        transform=transformCompose,
        batch_transforms=batch_transforms,
        sampling_rate=sr,  # Use passed sr
        mono=True,  # Data is now guaranteed mono
        device=target_device  # Pass the target device here
    )

    dataloader = {
        'predict':
            torch.utils.data.DataLoader(  # type: ignore
                dataset['predict'],
                shuffle=False,
                batch_size=16)
    }
    return dataloader


def setup(
        args: argparse.Namespace
) -> Tuple[torch.nn.Module, Dict, edansa.io.IO]:
    """
    Sets up the configuration, determines the target device, loads the model onto that device,
    and initializes the IO handler.
    """
    config = get_config(args)

    # --- Determine target device for loading model ---
    # This logic mirrors _setup_device_and_config but is needed *before* model loading
    if args.device:
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            logging.warning(
                f"User specified device '{args.device}' but CUDA not available. Using CPU for model loading."
            )
            device_str = 'cpu'
        elif args.device == 'cpu':
            device_str = 'cpu'
        elif args.device.startswith('cuda'):
            # Minimal validation for loading, more thorough check in main script if needed
            if ':' in args.device:
                try:
                    gpu_index = int(args.device.split(':')[1])
                    if gpu_index >= torch.cuda.device_count():
                        logging.warning(
                            f"CUDA device index {gpu_index} out of range for loading. Using default CUDA device."
                        )
                        device_str = 'cuda'
                    else:
                        device_str = args.device
                except (ValueError, IndexError):
                    logging.warning(
                        f"Invalid CUDA device format '{args.device}' for loading. Using default CUDA device."
                    )
                    device_str = 'cuda'
            else:
                device_str = 'cuda'  # Default cuda device
        else:
            logging.warning(
                f"Unrecognized device '{args.device}'. Using default (CUDA if available, else CPU) for loading."
            )
            device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        # Default logic if --device not specified
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

    map_location_device = torch.device(device_str)
    logging.info(
        f"Determined map_location for torch.load: {map_location_device}")

    # --- Load Model onto the determined device ---
    model_saved, config = get_model(
        config,
        category_count=len(config['target_taxo']),
        map_location_device=map_location_device)  # Pass determined device

    # --- Setup IO Handler ---
    # Output folder default handled in main script's __main__ block now
    output_folder = args.output_folder  # Use the path possibly set by default handling

    file_io = edansa.io.IO(
        excerpt_len=config['excerpt_length'],
        output_folder=output_folder,
    )

    # The main script will call _setup_device_and_config to finalize device setup
    # (like setting model.eval() and potentially moving if map_location somehow failed)
    # and update config['device'] for downstream use (e.g., dataloader).
    # We *don't* set config['device'] here definitively, let the main script do that.

    return model_saved, config, file_io


# %%

if __name__ == '__main__':
    # Create the parser using the function from inference module
    parser = inference.create_arg_parser()
    # Parse arguments defined in inference.py
    args = parser.parse_args()

    # Handle default output folder path creation *after* parsing
    if args.output_folder is None:
        output_path = Path("outputs")
        # The directory itself is created by the IO handler later
        args.output_folder = str(output_path.resolve())
        logging.info(
            f"Output folder not specified, using default: {args.output_folder}")

    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("Starting inference script.")

    # Call the main function from the inference module
    inference.main(args, setup, prepare_dataloader_from_audio_ins)
    logging.info("Inference script finished.")
