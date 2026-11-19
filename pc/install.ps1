<#
install.ps1  (ASCII only)
Run once on the home PC (the one where rclone is already set up with the "gdrive" remote):

  irm https://raw.githubusercontent.com/yaimamiyara-ui/boatrace-predictor/main/pc/install.ps1 | iex

It downloads fetch_day.ps1 into %LOCALAPPDATA%\boatrace-fetch, registers a daily task at 07:25
(runs later if the PC was off at that time), and starts it once now.
Uninstall:  Unregister-ScheduledTask -TaskName boatrace-fetch -Confirm:$false
#>
$ErrorActionPreference = "Stop"
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}
$root = Join-Path $env:LOCALAPPDATA "boatrace-fetch"
New-Item -ItemType Directory -Force -Path $root | Out-Null
$script = Join-Path $root "fetch_day.ps1"
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/yaimamiyara-ui/boatrace-predictor/main/pc/fetch_day.ps1" -OutFile $script

$rc = Get-Command rclone -ErrorAction SilentlyContinue
$rcPath = if ($rc) { $rc.Source } else {
    $f = $null
    foreach ($d in @($root, "$env:USERPROFILE\Downloads", "$env:USERPROFILE\Documents", "$env:USERPROFILE\Desktop", $env:ProgramFiles, "C:\rclone")) {
        if (Test-Path $d) { $f = Get-ChildItem -Path $d -Filter rclone.exe -Recurse -Depth 2 -ErrorAction SilentlyContinue | Select-Object -First 1; if ($f) { break } }
    }
    if ($f) { $f.FullName } else { $null }
}
if (-not $rcPath) { Write-Host "NG: rclone.exe was not found on this PC." -ForegroundColor Red; return }
$remotes = & $rcPath listremotes
if (-not ($remotes -contains "gdrive:")) { Write-Host "NG: rclone has no remote named gdrive on this PC." -ForegroundColor Red; return }
Write-Host "rclone: $rcPath"

$arg = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`" -Rclone `"$rcPath`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg
$trigger = New-ScheduledTaskTrigger -Daily -At "07:25"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 18) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "boatrace-fetch" -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName "boatrace-fetch"
Write-Host "OK: task 'boatrace-fetch' registered (daily 07:25) and started." -ForegroundColor Green
Write-Host "Log: $root\log"
