@echo off
setlocal

:: Setup MSVC Environment
set VCVARS="C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"

if not exist %VCVARS% (
    echo [ERROR]: vcvars64.bat not found at %VCVARS%
    exit /b 1
)

:: We must call vcvars64 inside the same shell context
call %VCVARS%

:: Now run NVCC with the unsupported compiler flag
:: -allow-unsupported-compiler bypasses the MSVC version check
nvcc -o sovereign_05.exe sovereign_05.cu -arch=sm_80 -O3 -Xcompiler=-Wall -allow-unsupported-compiler

if %errorlevel% neq 0 (
    echo [ERROR]: NVCC Compilation Failed
    exit /b %errorlevel%
)

echo [SUCCESS]: Sovereign 0.5 Unified Core Compiled.
echo [RUNNING]: Executing Phase 1 Baseline...
.\sovereign_05.exe
