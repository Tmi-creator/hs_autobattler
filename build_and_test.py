import subprocess, sys, os

os.environ["PATH"] = r"C:\msys64\mingw64\bin;" + os.environ["PATH"]

# Get pybind11 cmake dir
import pybind11
pb_dir = pybind11.get_cmake_dir()
print(f"pybind11 cmake dir: {pb_dir}")

cpp_dir = os.path.join(os.path.dirname(__file__), "cpp")
build_dir = os.path.join(cpp_dir, "build")

# Configure
print("=== CMake Configure ===")
r = subprocess.run([
    "cmake",
    "-G", "MinGW Makefiles",
    "-B", build_dir,
    "-S", cpp_dir,
    f"-Dpybind11_DIR={pb_dir}",
], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr)
    sys.exit(1)

# Build
print("=== CMake Build ===")
r = subprocess.run([
    "cmake", "--build", build_dir, "--config", "Release", "-j4"
], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr)
    sys.exit(1)

# Test import
print("=== Testing import ===")
sys.path.insert(0, build_dir)
os.add_dll_directory(r"C:\msys64\mingw64\bin")
import hs_engine_cpp
size = hs_engine_cpp.get_state_size()
print(f"CombatState size: {size} bytes")
print("SUCCESS!")
