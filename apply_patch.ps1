param(
    [string]$Target = (Get-Location).Path
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$patch = Join-Path $here 'bsaireader-gpt-sovits.patch'
$probe = Join-Path $Target 'GPT_SoVITS/TTS_infer_pack/TTS.py'

if (-not (Test-Path $probe)) {
    throw "Target does not look like a GPT-SoVITS-For-Intel checkout: $Target"
}

Push-Location $Target
try {
    & git -c core.autocrlf=false apply --check $patch
    if ($LASTEXITCODE -ne 0) { throw 'git apply --check failed.' }
    & git -c core.autocrlf=false apply $patch
    if ($LASTEXITCODE -ne 0) { throw 'git apply failed.' }
} finally {
    Pop-Location
}

Write-Host "Applied: $patch"
Write-Host "Optional runtime-state patch (not applied): $(Join-Path $here 'optional/weight.json.patch')"
