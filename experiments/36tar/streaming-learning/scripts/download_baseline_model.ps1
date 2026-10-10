$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$modelRoot = Join-Path $projectRoot 'models/Qwen3.5-0.8B'
$revision = '2fc06364715b967f1860aea9cf38778875588b17'
$endpoint = 'https://hf-mirror.com'
New-Item -ItemType Directory -Path $modelRoot -Force | Out-Null
$handler = [System.Net.Http.HttpClientHandler]::new()
$handler.UseProxy = $false
$client = [System.Net.Http.HttpClient]::new($handler)
$client.Timeout = [TimeSpan]::FromMinutes(15)
$client.DefaultRequestHeaders.UserAgent.ParseAdd('Qwen35-Baseline-Preparation')
$manifest = @()
try {
    $metadata = $client.GetStringAsync("$endpoint/api/models/Qwen/Qwen3.5-0.8B/revision/${revision}?blobs=true").GetAwaiter().GetResult() | ConvertFrom-Json
    if ($metadata.sha -ne $revision) { throw 'Model revision mismatch' }
    $metadata | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $modelRoot 'DOWNLOAD_METADATA.json') -Encoding UTF8
    $allowed = $metadata.siblings | Where-Object { $_.rfilename -match '^[^/]+\.(json|txt|jinja|safetensors)$' }
    # Download small configuration/tokenizer files first, then weights; no other model formats.
    $allowed = $allowed | Sort-Object @{Expression={ $_.rfilename.EndsWith('.safetensors') }},rfilename
    foreach ($entry in $allowed) {
        $name = $entry.rfilename
        if ($name -ne [System.IO.Path]::GetFileName($name)) { throw 'Unexpected nested path' }
        $target = Join-Path $modelRoot $name
        $partial = $target + '.partial'
        $expectedSize = if ($entry.lfs) { [long]$entry.lfs.size } else { [long]$entry.size }
        $expectedHash = if ($entry.lfs) { $entry.lfs.sha256 } else { $null }
        $validExisting = (Test-Path -LiteralPath $target -PathType Leaf) -and ((Get-Item -LiteralPath $target).Length -eq $expectedSize)
        if ($validExisting -and $expectedHash) {
            $validExisting = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -eq $expectedHash
        }
        if (-not $validExisting) {
            Write-Output "DOWNLOAD $name expected_bytes=$expectedSize"
            $response = $client.GetAsync("$endpoint/Qwen/Qwen3.5-0.8B/resolve/$revision/$name", [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
            $response.EnsureSuccessStatusCode() | Out-Null
            $inputStream = $response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
            $outputStream = [System.IO.File]::Open($partial,[System.IO.FileMode]::Create,[System.IO.FileAccess]::Write)
            $buffer = [byte[]]::new(1048576)
            $written = [long]0
            $timer = [System.Diagnostics.Stopwatch]::StartNew()
            $lastReport = [double]0
            try {
                while (($count = $inputStream.Read($buffer,0,$buffer.Length)) -gt 0) {
                    $outputStream.Write($buffer,0,$count)
                    $written += $count
                    if ($timer.Elapsed.TotalSeconds - $lastReport -ge 30) {
                        Write-Output "PROGRESS $name bytes=$written expected=$expectedSize"
                        $lastReport = $timer.Elapsed.TotalSeconds
                    }
                }
            } finally { $outputStream.Dispose(); $inputStream.Dispose(); $response.Dispose() }
            if ((Get-Item -LiteralPath $partial).Length -ne $expectedSize) { throw "Size mismatch: $name" }
            if ($expectedHash -and (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $expectedHash) { throw "SHA256 mismatch: $name" }
            Move-Item -LiteralPath $partial -Destination $target -Force
        }
        $manifest += [PSCustomObject]@{name=$name;bytes=(Get-Item -LiteralPath $target).Length;sha256=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash;expected_lfs_sha256=$expectedHash}
        [PSCustomObject]@{model='Qwen/Qwen3.5-0.8B';revision=$revision;endpoint=$endpoint;files=$manifest} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $modelRoot 'DOWNLOAD_MANIFEST.json') -Encoding UTF8
        Write-Output "VERIFIED $name"
    }
    Write-Output "PASS model preparation: $($manifest.Count) files verified, revision $revision"
} finally { $client.Dispose(); $handler.Dispose() }
