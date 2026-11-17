$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot 'vendor\pi05-probe-client\Scripts\python.exe'
& $taskPython -u (Join-Path $PSScriptRoot 'collect_so101_medicine.py')
if ($LASTEXITCODE -ne 0) { throw 'Collection failed. Check leader COM5, follower COM6, cameras, and calibration files.' }
