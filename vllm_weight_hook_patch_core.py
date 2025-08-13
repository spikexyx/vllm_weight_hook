# vllm_weight_hook_patch_core.py
'''
Usage: 
Use install_vllm_hook_patch.sh to install the VLLM patch.
Use uninstall_vllm_hook_patch.sh to remove the patch.
Or manually:
Put vllm_patch_loader.py & vllm_weight_hook_patch_core.py & vllm_injector.pth into the python site-packages directory of the target environment.
Use this command to find the site-packages directory:
python -c "import site; print(site.getsitepackages()[0])"
'''

import sys
import os
import fcntl
# import runpy
import json
import time
import torch
from typing import List, Tuple, Union, Optional
import atexit
import tempfile
import shutil

print(f"[VLLM_PATCH] Patch Module loaded in process: {os.getpid()}")
# ===================================================================
# All patching code for model runners to handle weight metadata saving
def _patched_acquire_weight_lock(self, timeout=10):
    """acquire weight metadata saving file lock"""
    temp_dir = tempfile.gettempdir()
    metadata_dir = os.path.join(temp_dir, "weights_metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    lock_file = os.path.join(metadata_dir, f"weight_saving_{parallel_state_module.get_tensor_model_parallel_rank()}.lock")

    try:
        self._lock_fd = os.open(lock_file, os.O_CREAT | os.O_WRONLY)
        start_time = time.time()

        while True:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # logger.info(f"Acquired weight saving lock for GPU {self.gpu_id}")
                return True
            except IOError:
                if time.time() - start_time > timeout:
                    # logger.error(f"Failed to acquire weight lock within {timeout} seconds")
                    os.close(self._lock_fd)
                    return False
                time.sleep(0.1)
    except Exception as e:
        # logger.error(f"Error acquiring weight lock: {e}")
        return False

def _patched_release_weight_lock(self):
    """release weight metadata saving file lock"""
    temp_dir = tempfile.gettempdir()
    metadata_dir = os.path.join(temp_dir, "weights_metadata")
    if hasattr(self, '_lock_fd'):
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            # delete lock file
            lock_file = os.path.join(metadata_dir, f"weight_saving_{parallel_state_module.get_tensor_model_parallel_rank()}.lock")
            if os.path.exists(lock_file):
                os.remove(lock_file)
            # logger.info(f"Released weight saving lock for GPU {self.gpu_id}")
        # except Exception as e:
            # logger.warning(f"Error releasing weight lock: {e}")
        finally:
            delattr(self, '_lock_fd')

# Weights_hook function 
def _patched_register_weight_hooks(self):
    # self.weight_infos = {}  # Save weight metadatas
    self._clear_old_weight_data()

    def tensor_hook(tensor: torch.Tensor, name: str):
        if tensor.is_cuda:
            self.weight_infos[name] = {
                "ptr": tensor.data_ptr(),
                "size": tensor.numel() * tensor.element_size(),
                # "actual_size": tensor.storage().size() * tensor.element_size(),
                "device": str(tensor.device),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape)
            }

    if not self._acquire_weight_lock():
        raise RuntimeError("Failed to acquire weight metadata update lock")

    # Register hooks to capture the initial state of model weights
    for name, param in self.model.named_parameters():
        tensor_hook(param, name)  # Capture parameter weights
    self._save_weight_meta()  # Save weight metadata to a local file
    self.total_weight_dict = self._calculate_device_weight_sizes(unit="GB")
    self._save_total_weight_meta()
    # self._merge_weights()  # Merge weights based on pointer continuity
    # self._save_merged_weight_meta()  # Save merged weight metadata to a local file
    self._release_weight_lock()

# Save the model weight metadata to a JSON file
def _patched_save_weight_meta(self):
    temp_dir = tempfile.gettempdir()
    metadata_dir = os.path.join(temp_dir, "weights_metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    meta_path = os.path.join(metadata_dir, f"weights_meta_{parallel_state_module.get_tensor_model_parallel_rank()}.json")
    # meta_path = f"weights_meta_{self.gpu_id}.json"
    try:
        with open(meta_path, 'w') as f:
            json.dump(self.weight_infos, f, indent=2)
        # logger.info(f"Save weight metadata to {meta_path}.")
    except IOError as e:
        # logger.error(f"Failed to save weight metadata to {meta_path}: {e}")
        raise

def _patched_save_total_weight_meta(self):
    # temp_dir = tempfile.gettempdir()
    # metadata_dir = os.path.join(temp_dir, "weights_metadata")
    total_metadata_dir = os.path.join(tempfile.gettempdir(), "total_weights_metadata")
    os.makedirs(total_metadata_dir, exist_ok=True)
    meta_path = os.path.join(total_metadata_dir, f"total_weight_meta_{parallel_state_module.get_tensor_model_parallel_rank()}.json")
    # meta_path = f"weights_meta_{self.gpu_id}.json"
    try:
        with open(meta_path, 'w') as f:
            json.dump(self.total_weight_dict, f, indent=2)
        # logger.info(f"Save total weight metadata to {meta_path}.")
    except IOError as e:
        # logger.error(f"Failed to save total weight metadata to {meta_path}: {e}")
        raise

def _patched_calculate_device_weight_sizes(self, unit: str = "bytes") -> dict:
    """Calculate the total size of weights per device in self.weight_infos.
    
    Args:
        unit (str): The unit to return the size in. 
                    Options: "bytes", "KB", "MB", "GB".
    
    Returns:
        dict: {device: total_size} where total_size is in the specified unit.
    """
    device_sizes = {}  # {device: total_size_in_bytes}

    for info in self.weight_infos.values():
        device = info["device"]
        size = info["size"]
        if device in device_sizes:
            device_sizes[device] += size
        else:
            device_sizes[device] = size

    unit = unit.upper()
    if unit == "KB":
        return {device: size / 1024 for device, size in device_sizes.items()}
    elif unit == "MB":
        return {device: size / (1024 ** 2) for device, size in device_sizes.items()}
    elif unit == "GB":
        return {device: size / (1024 ** 3) for device, size in device_sizes.items()}
    else:  # Default to bytes
        return device_sizes
    
def _patched_clear_old_weight_data(self):
    """
    Clear old weight information and metadata files
    """
    # Clear in-memory data
    if hasattr(self, 'weight_infos'):
        self.weight_infos.clear()
    else:
        self.weight_infos = {}

    if hasattr(self, 'total_weight_dict'):
        self.total_weight_dict.clear()
    else:
        self.total_weight_dict = {}

    # Remove old metadata files
    try:
        temp_dir = tempfile.gettempdir()
        metadata_dir = os.path.join(temp_dir, "weights_metadata")
        if os.path.exists(metadata_dir):
            shutil.rmtree(metadata_dir)
            print(f"[VLLM_PATCH_CORE] Global cleanup: removed {metadata_dir}")
        
        total_metadata_dir = os.path.join(temp_dir, "total_weights_metadata")
        if os.path.exists(total_metadata_dir):
            shutil.rmtree(total_metadata_dir)
            print(f"[VLLM_PATCH_CORE] Global cleanup: removed {total_metadata_dir}")

    except Exception as e:
        # logger.warning(f"Failed to clean old metadata files: {e}")
        return
    
# Global cleanup function
def _cleanup_all_metadata_files():
    """global cleanup metadata"""
    try:
        temp_dir = tempfile.gettempdir()
        
        metadata_dir = os.path.join(temp_dir, "weights_metadata")
        if os.path.exists(metadata_dir):
            shutil.rmtree(metadata_dir)
            print(f"[VLLM_PATCH_CORE] Global cleanup: removed {metadata_dir}")
        
        total_metadata_dir = os.path.join(temp_dir, "total_weights_metadata")
        if os.path.exists(total_metadata_dir):
            shutil.rmtree(total_metadata_dir)
            print(f"[VLLM_PATCH_CORE] Global cleanup: removed {total_metadata_dir}")
            
    except Exception as e:
        print(f"[VLLM_PATCH_CORE] Global cleanup error: {e}")

# ===================================================================
# Monkey patch the ModelRunner classes load_model method
def apply_vllm_v0_gpu_model_runner_patch():
    try:
        # Patch v0 GPUModelRunnerBase load_model
        from vllm.worker.model_runner import GPUModelRunnerBase

        _cleanup_all_metadata_files()

        # Register cleanup function in atexit
        atexit.register(_cleanup_all_metadata_files)

        GPUModelRunnerBase._acquire_weight_lock = _patched_acquire_weight_lock
        GPUModelRunnerBase._release_weight_lock = _patched_release_weight_lock
        GPUModelRunnerBase._register_weight_hooks = _patched_register_weight_hooks
        GPUModelRunnerBase._save_weight_meta = _patched_save_weight_meta
        GPUModelRunnerBase._save_total_weight_meta = _patched_save_total_weight_meta
        GPUModelRunnerBase._calculate_device_weight_sizes = _patched_calculate_device_weight_sizes
        GPUModelRunnerBase._clear_old_weight_data = _patched_clear_old_weight_data

        if not hasattr(GPUModelRunnerBase, '_original_load_model'):
            # print("[VLLM_PATCH_CORE] Start to patching GPUModelRunnerBase.load_model to handle weight metadata loading")
            GPUModelRunnerBase._original_load_model = GPUModelRunnerBase.load_model
            def patched_load_model(self):
                print("[VLLM_PATCH_CORE] Patched GPUModelRunnerBase.load_model to handle weight metadata loading")
                self._original_load_model()
                # Register hooks after model is loaded
                self._register_weight_hooks()
            GPUModelRunnerBase.load_model = patched_load_model
    except Exception as e:
        print(f"[VLLM_PATCH_CORE] Failed to apply GPUModelRunnerBase patches: {e}")
        return
    
def apply_vllm_v0_hpu_model_runner_patch():
    try:
        # Patch v0 hpu_model_runner load_model
        from vllm.worker.hpu_model_runner import HPUModelRunnerBase

        HPUModelRunnerBase._acquire_weight_lock = _patched_acquire_weight_lock
        HPUModelRunnerBase._release_weight_lock = _patched_release_weight_lock
        HPUModelRunnerBase._register_weight_hooks = _patched_register_weight_hooks
        HPUModelRunnerBase._save_weight_meta = _patched_save_weight_meta
        HPUModelRunnerBase._save_total_weight_meta = _patched_save_total_weight_meta
        HPUModelRunnerBase._calculate_device_weight_sizes = _patched_calculate_device_weight_sizes
        HPUModelRunnerBase._clear_old_weight_data = _patched_clear_old_weight_data

        if not hasattr(HPUModelRunnerBase, '_original_load_model'):
            # print("[VLLM_PATCH_CORE] Start to patching HPUModelRunnerBase.load_model to handle weight metadata loading")
            HPUModelRunnerBase._original_load_model = HPUModelRunnerBase.load_model
            def patched_load_model(self):
                print("[VLLM_PATCH_CORE] Patched HPUModelRunnerBase.load_model to handle weight metadata loading")
                self._original_load_model()
                # Register hooks after model is loaded
                self._register_weight_hooks()
            HPUModelRunnerBase.load_model = patched_load_model

    except Exception as e:
        print(f"[VLLM_PATCH_CORE] Failed to apply HPUModelRunner patches: {e}")
        return
    
def apply_vllm_v0_multi_step_neuron_model_runner_patch():
    try:
        # Patch v0 multi_step_neuron_model_runner load_model
        from vllm.worker.multi_step_neuron_model_runner import MultiStepNeuronModelRunner

        MultiStepNeuronModelRunner._acquire_weight_lock = _patched_acquire_weight_lock
        MultiStepNeuronModelRunner._release_weight_lock = _patched_release_weight_lock
        MultiStepNeuronModelRunner._register_weight_hooks = _patched_register_weight_hooks
        MultiStepNeuronModelRunner._save_weight_meta = _patched_save_weight_meta
        MultiStepNeuronModelRunner._save_total_weight_meta = _patched_save_total_weight_meta
        MultiStepNeuronModelRunner._calculate_device_weight_sizes = _patched_calculate_device_weight_sizes
        MultiStepNeuronModelRunner._clear_old_weight_data = _patched_clear_old_weight_data

        if not hasattr(MultiStepNeuronModelRunner, '_original_load_model'):
            # print("[VLLM_PATCH_CORE] Start to patching MultiStepNeuronModelRunner.load_model to handle weight metadata loading")
            MultiStepNeuronModelRunner._original_load_model = MultiStepNeuronModelRunner.load_model
            def patched_load_model(self):
                print("[VLLM_PATCH_CORE] Patched MultiStepNeuronModelRunner.load_model to handle weight metadata loading")
                self._original_load_model()
                # Register hooks after model is loaded
                self._register_weight_hooks()
            MultiStepNeuronModelRunner.load_model = patched_load_model

    except Exception as e:
        print(f"[VLLM_PATCH_CORE] Failed to apply MultiStepNeuronModelRunner patches: {e}")
        return
    
def apply_vllm_v0_multi_step_neuronx_distributed_model_runner_patch():
    try:
        # Patch v0 multi_step_neuronx_distributed_model_runner load_model
        from vllm.worker.multi_step_neuronx_distributed_model_runner import MultiStepNeuronxDistributedModelRunner

        MultiStepNeuronxDistributedModelRunner._acquire_weight_lock = _patched_acquire_weight_lock
        MultiStepNeuronxDistributedModelRunner._release_weight_lock = _patched_release_weight_lock
        MultiStepNeuronxDistributedModelRunner._register_weight_hooks = _patched_register_weight_hooks
        MultiStepNeuronxDistributedModelRunner._save_weight_meta = _patched_save_weight_meta
        MultiStepNeuronxDistributedModelRunner._save_total_weight_meta = _patched_save_total_weight_meta
        MultiStepNeuronxDistributedModelRunner._calculate_device_weight_sizes = _patched_calculate_device_weight_sizes
        MultiStepNeuronxDistributedModelRunner._clear_old_weight_data = _patched_clear_old_weight_data

        if not hasattr(MultiStepNeuronxDistributedModelRunner, '_original_load_model'):
            # print("[VLLM_PATCH_CORE] Start to patching MultiStepNeuronxDistributedModelRunner.load_model to handle weight metadata loading")
            MultiStepNeuronxDistributedModelRunner._original_load_model = MultiStepNeuronxDistributedModelRunner.load_model
            def patched_load_model(self):
                print("[VLLM_PATCH_CORE] Patched MultiStepNeuronxDistributedModelRunner.load_model to handle weight metadata loading")
                self._original_load_model()
                # Register hooks after model is loaded
                self._register_weight_hooks()
            MultiStepNeuronxDistributedModelRunner.load_model = patched_load_model

    except Exception as e:
        print(f"[VLLM_PATCH_CORE] Failed to apply MultiStepNeuronxDistributedModelRunner patches: {e}")
        return
    
def apply_vllm_v0_neuron_model_runner_patch():
    try:
        # Patch v0 neuron_model_runner load_model
        from vllm.worker.neuron_model_runner import NeuronModelRunner

        NeuronModelRunner._acquire_weight_lock = _patched_acquire_weight_lock
        NeuronModelRunner._release_weight_lock = _patched_release_weight_lock
        NeuronModelRunner._register_weight_hooks = _patched_register_weight_hooks
        NeuronModelRunner._save_weight_meta = _patched_save_weight_meta
        NeuronModelRunner._save_total_weight_meta = _patched_save_total_weight_meta
        NeuronModelRunner._calculate_device_weight_sizes = _patched_calculate_device_weight_sizes
        NeuronModelRunner._clear_old_weight_data = _patched_clear_old_weight_data   

        if not hasattr(NeuronModelRunner, '_original_load_model'):
            # print("[VLLM_PATCH_CORE] Start to patching NeuronModelRunner.load_model to handle weight metadata loading")
            NeuronModelRunner._original_load_model = NeuronModelRunner.load_model
            def patched_load_model(self):
                print("[VLLM_PATCH_CORE] Patched NeuronModelRunner.load_model to handle weight metadata loading")
                self._original_load_model()
                # Register hooks after model is loaded
                self._register_weight_hooks()
            NeuronModelRunner.load_model = patched_load_model

    except Exception as e:
        print(f"[VLLM_PATCH_CORE] Failed to apply NeuronModelRunner patches: {e}")
        return
    
def apply_vllm_v0_neuronx_distributed_model_runner_patch():
    try:
        # Patch v0 neuronx_distributed_model_runner load_model
        from vllm.worker.neuronx_distributed_model_runner import NeuronxDistributedModelRunner

        NeuronxDistributedModelRunner._acquire_weight_lock = _patched_acquire_weight_lock
        NeuronxDistributedModelRunner._release_weight_lock = _patched_release_weight_lock
        NeuronxDistributedModelRunner._register_weight_hooks = _patched_register_weight_hooks
        NeuronxDistributedModelRunner._save_weight_meta = _patched_save_weight_meta
        NeuronxDistributedModelRunner._save_total_weight_meta = _patched_save_total_weight_meta
        NeuronxDistributedModelRunner._calculate_device_weight_sizes = _patched_calculate_device_weight_sizes
        NeuronxDistributedModelRunner._clear_old_weight_data = _patched_clear_old_weight_data

        if not hasattr(NeuronxDistributedModelRunner, '_original_load_model'):
            # print("[VLLM_PATCH_CORE] Start to patching NeuronxDistributedModelRunner.load_model to handle weight metadata loading")
            NeuronxDistributedModelRunner._original_load_model = NeuronxDistributedModelRunner.load_model
            def patched_load_model(self):
                print("[VLLM_PATCH_CORE] Patched NeuronxDistributedModelRunner.load_model to handle weight metadata loading")
                self._original_load_model()
                # Register hooks after model is loaded
                self._register_weight_hooks()
            NeuronxDistributedModelRunner.load_model = patched_load_model

    except Exception as e:
        print(f"[VLLM_PATCH_CORE] Failed to apply NeuronxDistributedModelRunner patches: {e}")
        return
    
def apply_vllm_v0_xpu_model_runner_patch():
    try:
        # Patch v0 xpu_model_runner load_model
        from vllm.worker.xpu_model_runner import XPUModelRunner

        XPUModelRunner._acquire_weight_lock = _patched_acquire_weight_lock
        XPUModelRunner._release_weight_lock = _patched_release_weight_lock
        XPUModelRunner._register_weight_hooks = _patched_register_weight_hooks
        XPUModelRunner._save_weight_meta = _patched_save_weight_meta
        XPUModelRunner._save_total_weight_meta = _patched_save_total_weight_meta
        XPUModelRunner._calculate_device_weight_sizes = _patched_calculate_device_weight_sizes
        XPUModelRunner._clear_old_weight_data = _patched_clear_old_weight_data

        if not hasattr(XPUModelRunner, '_original_load_model'):
            # print("[VLLM_PATCH_CORE] Start to patching XPUModelRunner.load_model to handle weight metadata loading")
            XPUModelRunner._original_load_model = XPUModelRunner.load_model
            def patched_load_model(self):
                print("[VLLM_PATCH_CORE] Patched XPUModelRunner.load_model to handle weight metadata loading")
                self._original_load_model()
                # Register hooks after model is loaded
                self._register_weight_hooks()
            XPUModelRunner.load_model = patched_load_model

    except Exception as e:
        print(f"[VLLM_PATCH_CORE] Failed to apply XPUModelRunner patches: {e}")
        return
    
def apply_vllm_v0_tpu_model_runner_patch():
    try:
        # Patch v0 tpu_model_runner load_model
        from vllm.worker.tpu_model_runner import TPUModelRunner

        TPUModelRunner._acquire_weight_lock = _patched_acquire_weight_lock
        TPUModelRunner._release_weight_lock = _patched_release_weight_lock
        TPUModelRunner._register_weight_hooks = _patched_register_weight_hooks
        TPUModelRunner._save_weight_meta = _patched_save_weight_meta
        TPUModelRunner._save_total_weight_meta = _patched_save_total_weight_meta
        TPUModelRunner._calculate_device_weight_sizes = _patched_calculate_device_weight_sizes
        TPUModelRunner._clear_old_weight_data = _patched_clear_old_weight_data

        if not hasattr(TPUModelRunner, '_original_load_model'):
            # print("[VLLM_PATCH_CORE] Start to patching TPUModelRunner.load_model to handle weight metadata loading")
            TPUModelRunner._original_load_model = TPUModelRunner.load_model
            def patched_load_model(self):
                print("[VLLM_PATCH_CORE] Patched TPUModelRunner.load_model to handle weight metadata loading")
                self._original_load_model()
                # Register hooks after model is loaded
                self._register_weight_hooks()
            TPUModelRunner.load_model = patched_load_model

    except Exception as e:
        print(f"[VLLM_PATCH_CORE] Failed to apply TPUModelRunner patches: {e}")
        return

def apply_vllm_v1_gpu_model_runner_patch():
    try:
        # Patch v1 GPUModelRunnerBase load_model
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        GPUModelRunner._acquire_weight_lock = _patched_acquire_weight_lock
        GPUModelRunner._release_weight_lock = _patched_release_weight_lock
        GPUModelRunner._register_weight_hooks = _patched_register_weight_hooks
        GPUModelRunner._save_weight_meta = _patched_save_weight_meta
        GPUModelRunner._save_total_weight_meta = _patched_save_total_weight_meta
        GPUModelRunner._calculate_device_weight_sizes = _patched_calculate_device_weight_sizes
        GPUModelRunner._clear_old_weight_data = _patched_clear_old_weight_data

        if not hasattr(GPUModelRunner, '_original_load_model'):
            # print("[VLLM_PATCH_CORE] Start to patching GPUModelRunner.load_model to handle weight metadata loading")
            GPUModelRunner._original_load_model = GPUModelRunner.load_model
            def patched_load_model(self):
                print("[VLLM_PATCH_CORE] Patched GPUModelRunner.load_model to handle weight metadata loading")
                self._original_load_model()
                # Register hooks after model is loaded
                self._register_weight_hooks()
            GPUModelRunner.load_model = patched_load_model

    except Exception as e:
        print(f"[VLLM_PATCH_CORE] Failed to apply v1 GPUModelRunner patches: {e}")
        return
    
def apply_vllm_v1_tpu_model_runner_patch():
    try:
        # Patch v1 tpu_model_runner load_model
        from vllm.v1.worker.tpu_model_runner import TPUModelRunner

        TPUModelRunner._acquire_weight_lock = _patched_acquire_weight_lock
        TPUModelRunner._release_weight_lock = _patched_release_weight_lock
        TPUModelRunner._register_weight_hooks = _patched_register_weight_hooks
        TPUModelRunner._save_weight_meta = _patched_save_weight_meta
        TPUModelRunner._save_total_weight_meta = _patched_save_total_weight_meta
        TPUModelRunner._calculate_device_weight_sizes = _patched_calculate_device_weight_sizes
        TPUModelRunner._clear_old_weight_data = _patched_clear_old_weight_data

        if not hasattr(TPUModelRunner, '_original_load_model'):
            # print("[VLLM_PATCH_CORE] Start to patching TPUModelRunner.load_model to handle weight metadata loading")
            TPUModelRunner._original_load_model = TPUModelRunner.load_model
            def patched_load_model(self):
                print("[VLLM_PATCH_CORE] Patched TPUModelRunner.load_model to handle weight metadata loading")
                self._original_load_model()
                # Register hooks after model is loaded
                self._register_weight_hooks()
            TPUModelRunner.load_model = patched_load_model

    except Exception as e:
        print(f"[VLLM_PATCH_CORE] Failed to apply v1 TPUModelRunner patches: {e}")
        return
    
# ===================================================================
# Apply all patches for all types of model runners
# TODO: Confirm other model runners' patch methods
def apply_vllm_model_runner_patches():
    print(f"[PATCH] Applying model runner patches in process {os.getpid()}...")

    import vllm.distributed.parallel_state as parallel_state_module

    apply_vllm_v0_gpu_model_runner_patch()
    # apply_vllm_v0_hpu_model_runner_patch()
    # apply_vllm_v0_multi_step_neuron_model_runner_patch()
    # apply_vllm_v0_multi_step_neuronx_distributed_model_runner_patch()
    # apply_vllm_v0_neuron_model_runner_patch()
    # apply_vllm_v0_neuronx_distributed_model_runner_patch()
    # apply_vllm_v0_xpu_model_runner_patch()
    # apply_vllm_v0_tpu_model_runner_patch()

    apply_vllm_v1_gpu_model_runner_patch()
    # apply_vllm_v1_tpu_model_runner_patch()
    
