[CmdletBinding()]
param(
    [string]$TargetPlatform = "linux/amd64",
    [string]$ArtifactRoot = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DeploymentTemplate = Join-Path $ProjectRoot "deployment"
if ([string]::IsNullOrWhiteSpace($ArtifactRoot)) {
    $ArtifactRoot = Join-Path $ProjectRoot "runtime\artifacts"
}
New-Item -ItemType Directory -Force -Path $ArtifactRoot | Out-Null
$ArtifactRoot = (Resolve-Path -LiteralPath $ArtifactRoot).Path

$Version = (Get-Content (Join-Path $DeploymentTemplate "VERSION") -Raw).Trim()
$ApiVersion = (Get-Content (Join-Path $DeploymentTemplate "API_IMAGE_VERSION") -Raw).Trim()
$WorkerVersion = (Get-Content (Join-Path $DeploymentTemplate "WORKER_IMAGE_VERSION") -Raw).Trim()
$PlatformSlug = $TargetPlatform.Replace("/", "-")
$ApiImage = "specification-checker-api:$ApiVersion"
$WorkerImage = "specification-checker-worker:$WorkerVersion"
$Stage = Join-Path $ArtifactRoot "specification-checker-update-$Version-$PlatformSlug"
$ImageArchiveName = "specification-checker-app-images-$Version-$PlatformSlug.tar"
$ImageArchive = Join-Path $Stage "docker-images\$ImageArchiveName"
$PackageName = "specification-checker-update-$Version-$PlatformSlug.tar.gz"
$PackagePath = Join-Path $ArtifactRoot $PackageName

if ($TargetPlatform -ne "linux/amd64") {
    throw "This release is validated only for linux/amd64"
}
if (Test-Path -LiteralPath $Stage) { throw "Refusing to overwrite: $Stage" }
if (Test-Path -LiteralPath $PackagePath) { throw "Refusing to overwrite: $PackagePath" }

$dockerCandidates = @(
    (Get-Command docker -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1),
    (Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\resources\bin\docker.exe"),
    "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
if (-not $dockerCandidates) { throw "Docker CLI is not installed" }
$Docker = $dockerCandidates[0]

function Invoke-Docker {
    & $Docker @args
    if ($LASTEXITCODE -ne 0) { throw "docker command failed: $($args -join ' ')" }
}

Invoke-Docker info | Out-Null
Invoke-Docker compose version | Out-Null

$gitCommit = $null
try {
    $gitCommit = (& git -c "safe.directory=$($ProjectRoot.Replace('\','/'))" rev-parse HEAD 2>$null).Trim()
    if ($LASTEXITCODE -ne 0) { $gitCommit = $null }
} catch { $gitCommit = $null }
$revision = if ($gitCommit) { $gitCommit } else { "unavailable-uncommitted-worktree" }

Write-Host "Building $ApiImage"
Invoke-Docker buildx build --platform $TargetPlatform --load --provenance=false --sbom=false --progress plain `
    --build-arg "APP_VERSION=$ApiVersion" --build-arg "VCS_REF=$revision" `
    --file (Join-Path $ProjectRoot "docker\Dockerfile.api") --tag $ApiImage $ProjectRoot

Write-Host "Building $WorkerImage"
Invoke-Docker buildx build --platform $TargetPlatform --load --provenance=false --sbom=false --progress plain `
    --build-arg "APP_VERSION=$WorkerVersion" --build-arg "VCS_REF=$revision" `
    --file (Join-Path $ProjectRoot "docker\Dockerfile.worker") --tag $WorkerImage $ProjectRoot

Write-Host "Validating image separation"
Invoke-Docker run --rm --pull never --platform $TargetPlatform --entrypoint python $ApiImage -c `
    "import importlib.util; assert importlib.util.find_spec('torch') is None; assert importlib.util.find_spec('transformers') is None"
Invoke-Docker run --rm --pull never --platform $TargetPlatform --entrypoint python $WorkerImage -c `
    "import torch, transformers; print('torch=' + torch.__version__ + ' transformers=' + transformers.__version__)"
foreach ($image in @($ApiImage, $WorkerImage)) {
    Invoke-Docker run --rm --pull never --platform $TargetPlatform --entrypoint sh $image -c `
        "test ! -e /app/tests && test ! -e /app/testdata && test ! -e /app/.git"
}

New-Item -ItemType Directory -Force -Path (Join-Path $Stage "docker-images") | Out-Null
Copy-Item (Join-Path $DeploymentTemplate "compose.production.yml") $Stage
Copy-Item (Join-Path $DeploymentTemplate ".env.production.example") $Stage
Copy-Item (Join-Path $DeploymentTemplate "VERSION") $Stage
Copy-Item (Join-Path $DeploymentTemplate "API_IMAGE_VERSION") $Stage
Copy-Item (Join-Path $DeploymentTemplate "WORKER_IMAGE_VERSION") $Stage
Copy-Item (Join-Path $DeploymentTemplate "README_UPDATE.md") $Stage
Copy-Item (Join-Path $DeploymentTemplate "scripts") $Stage -Recurse

Write-Host "Saving only the changed API and worker images"
Invoke-Docker save --output $ImageArchive $ApiImage $WorkerImage
if ((Get-Item $ImageArchive).Length -le 0) { throw "docker save produced an empty archive" }
Invoke-Docker load --input $ImageArchive

$imageEntries = foreach ($tag in @($ApiImage, $WorkerImage)) {
    $inspection = (& $Docker image inspect $tag | ConvertFrom-Json)[0]
    [ordered]@{
        tag = $tag
        id = $inspection.Id
        size_bytes = [int64]$inspection.Size
        platform = $TargetPlatform
    }
}
$imageHash = (Get-FileHash $ImageArchive -Algorithm SHA256).Hash.ToLowerInvariant()
$manifest = [ordered]@{
    schema_version = 1
    update_version = $Version
    build_date_utc = [DateTime]::UtcNow.ToString("o")
    git_commit = $gitCommit
    target_platform = $TargetPlatform
    images = $imageEntries
    worker_count = 2
    gpu_device_ids = @("0", "1")
    inference_path = "verified-local-safe-only"
    image_archive = [ordered]@{
        path = "docker-images/$ImageArchiveName"
        sha256 = $imageHash
        size_bytes = [int64](Get-Item $ImageArchive).Length
        docker_load_verified = $true
    }
    preserves = @("PostgreSQL image and volume", "Redis image and volume", "jobs_data", "MonkeyOCR model", ".env.production")
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText((Join-Path $Stage "manifest.json"), ($manifest | ConvertTo-Json -Depth 8), $utf8)

$checksumFiles = Get-ChildItem $Stage -Recurse -File | Where-Object Name -ne "checksums.sha256"
$checksumLines = foreach ($file in $checksumFiles) {
    $relative = $file.FullName.Substring($Stage.Length + 1).Replace("\", "/")
    "$((Get-FileHash $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant())  $relative"
}
[IO.File]::WriteAllLines((Join-Path $Stage "checksums.sha256"), $checksumLines, $utf8)

$temporaryEnv = Join-Path $Stage ".env.production"
Copy-Item (Join-Path $Stage ".env.production.example") $temporaryEnv
try {
    $config = & $Docker compose --env-file $temporaryEnv -f (Join-Path $Stage "compose.production.yml") config
    if ($LASTEXITCODE -ne 0) { throw "Standalone Compose config validation failed" }
    $configText = $config -join "`n"
    if ($configText -match "(?m)^\s*build:") { throw "Deployment Compose contains build:" }
    if ($configText -notmatch "worker-gpu-0" -or $configText -notmatch "worker-gpu-1") { throw "Both GPU workers are not configured" }
} finally {
    Remove-Item $temporaryEnv -Force -ErrorAction SilentlyContinue
}

$Python = Join-Path $ProjectRoot ".venv-gpu\Scripts\python.exe"
& $Python (Join-Path $ProjectRoot "tools\create_deployment_archive.py") $Stage $PackagePath
if ($LASTEXITCODE -ne 0) { throw "Could not create update archive" }
$packageHash = (Get-FileHash $PackagePath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText("$PackagePath.sha256", "$packageHash  $PackageName`n", $utf8)

Write-Host "UPDATE BUILD COMPLETE"
Write-Host "Package: $PackagePath"
Write-Host "Package SHA-256: $packageHash"
Write-Host "Image archive SHA-256: $imageHash"
