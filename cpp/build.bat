@echo off
set PATH=C:\msys64\mingw64\bin;%PATH%

echo === Configuring with CMake ===
cmake -G "MinGW Makefiles" -B build -S . -Dpybind11_DIR=%pybind11_DIR%

if %errorlevel% neq 0 (
    echo CMake configure failed!
    exit /b 1
)

echo === Building ===
cmake --build build --config Release -j4

if %errorlevel% neq 0 (
    echo Build failed!
    exit /b 1
)

echo === Build successful! ===
dir /b build\*.pyd 2>nul
dir /b build\*.dll 2>nul
