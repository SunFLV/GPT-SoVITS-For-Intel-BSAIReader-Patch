@echo off
chcp 65001 > nul
cd /d "%~dp0"

set "PYTHON=venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"

if not exist "%PYTHON%" (
    echo [ERROR] venv not found: %PYTHON%
    echo         run install.ps1 first
    pause
    exit /b 1
)

echo ============================================================
echo   GPT-SoVITS Multi-route TTS API  (process pool)
echo ------------------------------------------------------------
echo   default : 4 workers / port 9881 / IPv6 dual-stack / warmup
echo   public  : go-pool-api.bat --api-key auto --open-firewall
echo   examples: go-pool-api.bat --pool-size 3 --port 9882
echo             go-pool-api.bat --no-ipv6   (IPv4 only)
echo             go-pool-api.bat --no-warm
echo ------------------------------------------------------------
echo   endpoints:
echo     POST /                 single sentence  -^> audio/wav
echo     POST /tts_batch        texts[] -^> {sample_rate, wavs[b64]}
echo     POST /tts_batch_stream texts[] -^> NDJSON {index, wav}
echo     GET  /health  /docs    status / OpenAPI
echo ------------------------------------------------------------
echo   NOTE: run this in a normal console window. Launching it
echo         detached (scheduled task / redirected Start-Process)
echo         makes ROCm workers fail to load.
echo ============================================================
echo.

"%PYTHON%" api_pool.py %*

if errorlevel 1 (
    echo.
    echo [exited with error] see messages above
    pause
)
