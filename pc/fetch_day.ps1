<#
fetch_day.ps1  (ASCII only: Windows PowerShell 5.1 reads BOM-less files as ANSI)

Runs on the home PC once a day (Task Scheduler, 07:25 JST). Fetches a small, fixed set of
pages from the official boat race site and uploads the raw files to Google Drive with rclone.
GitHub Actions never touches the official site; it only reads these raw files.

Per day (about 85 requests, hard cap -MaxRequests):
  - program (B) / result (K) LZH files missing for the last few days
  - index page + one racelist page per venue (deadlines)
  - for the ~20 races in watchlist.csv (chosen by GitHub): beforeinfo + odds3t 9 min before
    the deadline, and odds3t again 25 min after the deadline (final odds)

Rules: at least 3 seconds between requests, no retries, no parallel requests.
On HTTP 403 or 429 the script stops at once and does not run again for 48 hours.
The default PowerShell User-Agent is used as is.
#>
param(
    [string]$Date = "",
    [int]$MaxRequests = 120,
    [int]$CatchupDays = 3,
    [int]$MinIntervalSec = 3,
    [string]$Rclone = "",
    [string]$Remote = "gdrive:boatrace-predictor/data/raw",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

$Root = Join-Path $env:LOCALAPPDATA "boatrace-fetch"
$Day = if ($Date) { [datetime]::ParseExact($Date, "yyyy-MM-dd", $null) } else { (Get-Date).Date }
$Ds = $Day.ToString("yyyy-MM-dd")
$Hd = $Day.ToString("yyyyMMdd")
$Work = Join-Path $Root $Ds
$LogFile = Join-Path $Root ("log\" + $Ds + ".log")
$BlockFile = Join-Path $Root "blocked_until.txt"
New-Item -ItemType Directory -Force -Path $Work, (Join-Path $Root "log"), (Join-Path $Root "lzh") | Out-Null

$script:Requests = 0
$script:LastRequest = [datetime]::MinValue

function Log([string]$msg) {
    $line = "{0} {1}" -f (Get-Date).ToString("HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

function Find-Rclone {
    if ($Rclone) { return $Rclone }
    $c = Get-Command rclone -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    $dirs = @($Root, (Join-Path $env:USERPROFILE "Downloads"), (Join-Path $env:USERPROFILE "Documents"),
              (Join-Path $env:USERPROFILE "Desktop"), $env:ProgramFiles, "C:\rclone")
    foreach ($d in $dirs) {
        if (-not $d -or -not (Test-Path $d)) { continue }
        $f = Get-ChildItem -Path $d -Filter rclone.exe -Recurse -Depth 2 -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($f) { return $f.FullName }
    }
    return $null
}

# Native commands: stderr must not become a terminating error under ErrorActionPreference=Stop (PS 5.1)
function Invoke-Rclone {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { $o = & $script:RcloneExe @args 2>&1 | ForEach-Object { "$_" } } finally { $ErrorActionPreference = $old }
    $script:RcloneExit = $LASTEXITCODE
    return $o
}

function Upload([string]$local, [string]$remoteRel) {
    if ($DryRun) { Log "DRYRUN upload $remoteRel"; return }
    Invoke-Rclone copyto $local "$Remote/$remoteRel" | Out-Null
    if ($script:RcloneExit -ne 0) { Log "WARN upload failed $remoteRel (rclone exit $($script:RcloneExit))" }
}

function Finish([int]$code) {
    Log "END requests=$($script:Requests) code=$code"
    Upload $LogFile ("log/" + $Ds + ".log")
    if ($script:Mutex) { try { $script:Mutex.ReleaseMutex() } catch {} }
    exit $code
}

# Returns "ok", "notfound" or "error". Exits the script on 403 / 429 / request cap.
function Fetch([string]$url, [string]$outFile) {
    if ($script:Requests -ge $MaxRequests) { Log "STOP request cap $MaxRequests reached"; Finish 3 }
    $wait = $MinIntervalSec - ((Get-Date) - $script:LastRequest).TotalSeconds
    if ($wait -gt 0) { Start-Sleep -Milliseconds ([int]($wait * 1000)) }
    $script:Requests++
    $script:LastRequest = Get-Date
    if ($DryRun) { Log "DRYRUN GET $url"; return "ok" }
    $status = 0
    try {
        Invoke-WebRequest -Uri $url -OutFile $outFile -UseBasicParsing -TimeoutSec 30
        Log "GET 200 $url"
        return "ok"
    } catch {
        $resp = $_.Exception.Response
        if ($resp) { try { $status = [int]$resp.StatusCode } catch {} }
        if (Test-Path $outFile) { Remove-Item $outFile -Force -ErrorAction SilentlyContinue }
        Log ("GET {0} {1} {2}" -f $status, $url, $_.Exception.Message)
    }
    if ($status -eq 403 -or $status -eq 429) {
        (Get-Date).AddHours(48).ToString("o") | Set-Content -Path $BlockFile -Encoding ASCII
        Log "STOP the site refused access ($status). No more requests for 48 hours."
        Finish 2
    }
    if ($status -eq 404) { return "notfound" }
    return "error"
}

function Wait-Until([datetime]$t) {
    while ((Get-Date) -lt $t) {
        $s = ($t - (Get-Date)).TotalSeconds
        Start-Sleep -Seconds ([int][math]::Max(1, [math]::Min(60, $s)))
    }
}

# ---- start ----
$created = $false
$script:Mutex = New-Object System.Threading.Mutex($true, "boatrace-fetch-day", [ref]$created)
if (-not $created) { Write-Host "already running"; exit 0 }

Log "START date=$Ds dryrun=$DryRun max=$MaxRequests"
if (Test-Path $BlockFile) {
    $until = [datetime]::Parse((Get-Content $BlockFile -Raw).Trim(), $null, [Globalization.DateTimeStyles]::RoundtripKind)
    if ((Get-Date) -lt $until) { Log "blocked until $until : nothing to do"; Finish 2 }
    Remove-Item $BlockFile -Force
}
$script:RcloneExe = Find-Rclone
if (-not $script:RcloneExe) { Log "ERROR rclone.exe not found"; Finish 1 }
Log "rclone=$($script:RcloneExe)"

# ---- 1. LZH files (program B up to today, results K up to yesterday) ----
function Get-Lzh {
    $have = @()
    $out = Invoke-Rclone lsf "$Remote/lzh"
    if ($script:RcloneExit -eq 0 -and $out) { $have = @($out | ForEach-Object { $_.Trim() }) }
    for ($i = $CatchupDays; $i -ge 0; $i--) {
        $d = $Day.AddDays(-$i)
        foreach ($kind in @("K", "B")) {
            if ($kind -eq "K" -and $i -eq 0) { continue }
            $name = "{0}{1}.lzh" -f $kind.ToLower(), $d.ToString("yyMMdd")
            if ($have -contains $name) { continue }
            $url = "https://www1.mbrace.or.jp/od2/{0}/{1}/{2}" -f $kind, $d.ToString("yyyyMM"), $name
            $local = Join-Path $Root ("lzh\" + $name)
            if ((Fetch $url $local) -eq "ok" -and -not $DryRun) {
                if ((Get-Item $local).Length -gt 200) { Upload $local ("lzh/" + $name) } else { Log "WARN too small $name" }
            }
        }
    }
}
Get-Lzh

# ---- 2. index + racelist (deadlines) ----
$base = "https://www.boatrace.jp/owpc/pc/race"
$index = Join-Path $Work "index.html"
if (-not (Test-Path $index)) {
    if ((Fetch "$base/index?hd=$Hd" $index) -eq "ok") { Upload $index "live/$Ds/index.html" }
}
$venues = @()
if (Test-Path $index) {
    $html = [IO.File]::ReadAllText($index)
    $venues = @([regex]::Matches($html, 'raceindex\?jcd=(\d{2})') | ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique)
}
Log "venues=$($venues -join ',')"
foreach ($jcd in $venues) {
    $f = Join-Path $Work "racelist_$jcd.html"
    if (Test-Path $f) { continue }
    if ((Fetch "$base/racelist?rno=1&jcd=$jcd&hd=$Hd" $f) -eq "ok") { Upload $f "live/$Ds/racelist_$jcd.html" }
}
if ($DryRun) { Log "DRYRUN stops before waiting for the watch list"; Finish 0 }
if ($venues.Count -eq 0) { Log "no venues today"; Finish 0 }

# ---- 3. wait for the watch list made by GitHub ----
$watch = Join-Path $Work "watchlist.csv"
$limit = (Get-Date).AddMinutes(150)
while (-not (Test-Path $watch) -and (Get-Date) -lt $limit) {
    Invoke-Rclone copyto "$Remote/live/$Ds/watchlist.csv" $watch | Out-Null
    if (-not (Test-Path $watch)) { Start-Sleep -Seconds 90 }
}
if (-not (Test-Path $watch)) { Log "no watch list arrived"; Finish 0 }
Get-Lzh   # second try for files that were not published yet at 07:25
$races = @(Import-Csv $watch)
Log "watch races=$($races.Count)"

# ---- 4. timed events ----
$events = @()
foreach ($r in $races) {
    $jcd = ([string]$r.jcd).PadLeft(2, "0")
    $rno = [int]$r.rno
    $hm = ([string]$r.deadline).Split(":")
    $dl = $Day.AddHours([int]$hm[0]).AddMinutes([int]$hm[1])
    $events += [pscustomobject]@{ At = $dl.AddMinutes(-9); Kind = "pre"; Jcd = $jcd; Rno = $rno; Deadline = $dl }
    $events += [pscustomobject]@{ At = $dl.AddMinutes(25); Kind = "final"; Jcd = $jcd; Rno = $rno; Deadline = $dl }
}
foreach ($e in ($events | Sort-Object At)) {
    $q = "rno=$($e.Rno)&jcd=$($e.Jcd)&hd=$Hd"
    if ($e.Kind -eq "pre") {
        $bf = Join-Path $Work "beforeinfo_$($e.Jcd)_$($e.Rno).html"
        $of = Join-Path $Work "odds3t_$($e.Jcd)_$($e.Rno).html"
        if (Test-Path $of) { continue }
        if ((Get-Date) -gt $e.Deadline.AddMinutes(-1)) { Log "skip pre $($e.Jcd)-$($e.Rno) (too late)"; continue }
        Wait-Until $e.At
        if ((Fetch "$base/beforeinfo?$q" $bf) -eq "ok") { Upload $bf "live/$Ds/beforeinfo_$($e.Jcd)_$($e.Rno).html" }
        if ((Fetch "$base/odds3t?$q" $of) -eq "ok") { Upload $of "live/$Ds/odds3t_$($e.Jcd)_$($e.Rno).html" }
    } else {
        $ff = Join-Path $Work "odds3tfinal_$($e.Jcd)_$($e.Rno).html"
        if (Test-Path $ff) { continue }
        Wait-Until $e.At
        if ((Fetch "$base/odds3t?$q" $ff) -eq "ok") { Upload $ff "live/$Ds/odds3tfinal_$($e.Jcd)_$($e.Rno).html" }
    }
}

# keep 14 days of local files
Get-ChildItem -Path $Root -Directory | Where-Object { $_.Name -match '^\d{4}-\d{2}-\d{2}$' -and $_.LastWriteTime -lt (Get-Date).AddDays(-14) } |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Finish 0
