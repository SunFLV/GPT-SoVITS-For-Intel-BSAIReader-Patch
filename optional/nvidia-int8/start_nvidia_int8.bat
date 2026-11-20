@echo off
setlocal EnableExtensions
rem ============================================================
rem  v5Turbo INT8 NVIDIA pool (P106-90 3G)
rem  Runs from the backend root. Keeps the Intel/XPU runtime intact.
rem ============================================================
cd /d "%~dp0"

set "BSAI_BACKEND_DIR=%~dp0"
set "BSAI_INT8_WEIGHTS=1"
set "BSAI_SERVER_NAME=v5Turbo_INT8 NVIDIA (P106)"
set "PY=%~dp0runtime_cuda\python.exe"
if not exist "%PY%" (
  echo [ERROR] CUDA runtime not found: %PY%
  pause
  exit /b 1
)

set "PATH=%~dp0runtime_cuda\Library\bin;%~dp0runtime_cuda;%~dp0runtime_cuda\Scripts;%PATH%"
set "PYTHONPATH=%~dp0;%~dp0GPT_SoVITS"
set "PYTHONMALLOC=malloc"
set "NO_PROXY=127.0.0.1,localhost,::1"
set "no_proxy=127.0.0.1,localhost,::1"
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

echo =============================================
echo   GPT-SoVITS v5Turbo INT8 / NVIDIA P106-90
echo   Backend: %BSAI_BACKEND_DIR%
echo   Pool:    http://127.0.0.1:9891
echo =============================================

"%PY%" "%~dp0api_pool.py" --config-file "%~dp0config\pool_api_nvidia_int8.json"

echo.
echo NVIDIA INT8 pool stopped.
pause
