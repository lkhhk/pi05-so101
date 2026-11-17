param(
    [switch]$Synthetic,
    [string]$RobotPort = 'COM6',
    [int]$Camera = 2,
    [int]$WristCamera = 0,
    [string]$Prompt = 'Pick up the block and place it in the box.'
)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot 'vendor\pi05-probe-client\Scripts\python.exe'
$taskOutput = Join-Path $taskRoot ('results\so101-pi05-probe-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '.npz')
$taskArgs = @((Join-Path $PSScriptRoot 'so101_pi05_probe.py'), '--robot-port', $RobotPort,
              '--camera', "$Camera", '--wrist-camera', "$WristCamera", '--prompt', $Prompt, '--output', $taskOutput)
if ($Synthetic) { $taskArgs += '--synthetic' }
& $taskPython @taskArgs
if ($LASTEXITCODE -ne 0) { throw "Probe failed with exit code $LASTEXITCODE" }
