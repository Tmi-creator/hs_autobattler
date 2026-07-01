import os
import subprocess
import time
import torch
import multiprocessing

# Worker thunk representing the environment subprocess (like in make_env)
def worker_thunk_before():
    # DO NOT clear CUDA_VISIBLE_DEVICES
    import torch
    # Initialize CUDA context by checking availability or doing a dummy operation
    _ = torch.cuda.is_available()
    if torch.cuda.is_available():
        x = torch.zeros(1, device="cuda")
    time.sleep(8)

def worker_thunk_after():
    # CUDA_VISIBLE_DEVICES was already cleared in parent process at spawn time
    import torch
    # Initialize CUDA context (should remain CPU-only and not touch VRAM)
    _ = torch.cuda.is_available()
    time.sleep(8)

def print_gpu_memory(msg):
    print(f"\n==========================================================")
    print(f" {msg}")
    print(f"==========================================================")
    try:
        out = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        print(out)
    except Exception as e:
        print(f"Could not run nvidia-smi: {e}")

if __name__ == "__main__":
    # Ensure spawn start method to match AsyncVectorEnv
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Initialize main PyTorch CUDA
    print(f"Main process CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        x = torch.zeros(1, device="cuda")
    
    print_gpu_memory("Initial VRAM usage")
    
    # ----------------------------------------------------
    # Case 1: BEFORE (no isolation)
    # ----------------------------------------------------
    print("\n>>> Starting Case 1: Spawning 12 subprocesses WITHOUT isolation...")
    processes_before = []
    for i in range(12):
        p = multiprocessing.Process(target=worker_thunk_before)
        p.start()
        processes_before.append(p)
    
    time.sleep(5)
    print_gpu_memory("VRAM usage DURING Case 1 (No isolation, 12 envs)")
    
    print("Cleaning up Case 1 processes...")
    for p in processes_before:
        p.terminate()
        p.join()
        
    time.sleep(2)
    print_gpu_memory("VRAM usage AFTER cleaning up Case 1")
    
    # ----------------------------------------------------
    # Case 2: AFTER (with isolation)
    # ----------------------------------------------------
    # Spawning subprocesses WITH parent environment isolation
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    processes_after = []
    for i in range(12):
        p = multiprocessing.Process(target=worker_thunk_after)
        p.start()
        processes_after.append(p)

    # Restore GPU for the parent process
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
        
    time.sleep(5)
    print_gpu_memory("VRAM usage DURING Case 2 (With isolation, 12 envs)")
    
    print("Cleaning up Case 2 processes...")
    for p in processes_after:
        p.terminate()
        p.join()
        
    print("\nFinished VRAM isolation test successfully.")
