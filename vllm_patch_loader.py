# vllm_patch_loader.py
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
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import ModuleSpec

import vllm_weight_hook_patch_core

TARGET_MODULES = {
    "vllm.worker.model_runner",
    "vllm.worker.hpu_model_runner",
    "vllm.worker.multi_step_neuron_model_runner",
    "vllm.worker.multi_step_neuronx_distributed_model_runner",
    "vllm.worker.neuron_model_runner",
    "vllm.worker.neuronx_distributed_model_runner",
    "vllm.worker.xpu_model_runner",
    "vllm.worker.tpu_model_runner",
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.worker.tpu_model_runner"
}

_patch_applied = False

class VllmPatcherLoader(Loader):
    def __init__(self, original_loader):
        self.original_loader = original_loader

    def exec_module(self, module):
        self.original_loader.exec_module(module)

        global _patch_applied
        if not _patch_applied:
            print(f"[VLLM_PATCH_LOADER] Target module '{module.__name__}' loaded. Applying patches...")
            try:
                vllm_weight_hook_patch_core.apply_vllm_model_runner_patches()
                _patch_applied = True
                print(f"[VLLM_PATCH_LOADER] Patches applied successfully in process {vllm_weight_hook_patch_core.os.getpid()}.")
            except Exception as e:
                print(f"[VLLM_PATCH_LOADER] Error applying patches: {e}", file=sys.stderr)


class VllmPatcherFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in TARGET_MODULES or _patch_applied:
            return None

        original_finder = self
        for finder in sys.meta_path:
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec:
                spec.loader = VllmPatcherLoader(spec.loader)
                return spec
        return None

sys.meta_path.insert(0, VllmPatcherFinder())

print(f"[VLLM_PATCH_LOADER] VLLM patch loader initialized in process {vllm_weight_hook_patch_core.os.getpid()}. Waiting for target module import...")
