[CmdletBinding()]
param(
    [string]$TargetPlatform = "linux/amd64",
    [string]$ArtifactRoot = "",
    [switch]$WorkerOnly
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$TemplateDeploymentRoot = Join-Path $ProjectRoot "deployment"
if ([string]::IsNullOrWhiteSpace($ArtifactRoot)) {
    $ArtifactRoot = $ProjectRoot
} else {
    New-Item -ItemType Directory -Force -Path $ArtifactRoot | Out-Null
    $ArtifactRoot = (Resolve-Path -LiteralPath $ArtifactRoot).Path
}
$DeploymentRoot = if ($ArtifactRoot -eq $ProjectRoot) {
    $TemplateDeploymentRoot
} else {
    Join-Path $ArtifactRoot "deployment"
}
if ($DeploymentRoot -ne $TemplateDeploymentRoot) {
    if (Test-Path -LiteralPath $DeploymentRoot) {
        throw "Refusing to overwrite existing staged deployment: $DeploymentRoot"
    }
    Copy-Item -LiteralPath $TemplateDeploymentRoot -Destination $DeploymentRoot -Recurse
}
$Version = (Get-Content -LiteralPath (Join-Path $DeploymentRoot "VERSION") -Raw).Trim()
$ApiVersion = (Get-Content -LiteralPath (Join-Path $DeploymentRoot "API_IMAGE_VERSION") -Raw).Trim()
$WorkerVersion = (Get-Content -LiteralPath (Join-Path $DeploymentRoot "WORKER_IMAGE_VERSION") -Raw).Trim()
$PlatformSlug = $TargetPlatform.Replace("/", "-")
$ApiImage = "specification-checker-api:$ApiVersion"
$WorkerImage = "specification-checker-worker:$WorkerVersion"
$PostgresImage = "postgres:18.4-alpine"
$RedisImage = "redis:7.4.10-alpine"
$ImageArchiveName = "specification-checker-images-$Version-$PlatformSlug.tar"
$ImageArchive = Join-Path $DeploymentRoot "docker-images\$ImageArchiveName"
$PackageName = "deployment-package-$Version-$PlatformSlug.tar.gz"
$PackagePath = Join-Path $ArtifactRoot $PackageName

if ($TargetPlatform -ne "linux/amd64") {
    throw "This release is validated only for linux/amd64. Revalidate CUDA/PyTorch before changing TARGET_PLATFORM."
}

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

$fingerprintRoots = @("app", "config", "alembic", "docker")
$fingerprintFiles = foreach ($root in $fingerprintRoots) {
    Get-ChildItem -LiteralPath (Join-Path $ProjectRoot $root) -Recurse -File
}
$fingerprintFiles += Get-Item -LiteralPath (Join-Path $ProjectRoot "pyproject.toml")
$fingerprintFiles += Get-Item -LiteralPath (Join-Path $ProjectRoot "requirements.txt")
$fingerprintFiles += Get-Item -LiteralPath (Join-Path $ProjectRoot "requirements-api.txt")
$fingerprintFiles += Get-Item -LiteralPath (Join-Path $ProjectRoot "requirements-monkeyocr.txt")
$fingerprintText = ($fingerprintFiles | Sort-Object FullName | ForEach-Object {
    $relative = $_.FullName.Substring($ProjectRoot.Length + 1).Replace("\", "/")
    "$(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256 | Select-Object -ExpandProperty Hash)  $relative"
}) -join "`n"
$sha = [System.Security.Cryptography.SHA256]::Create()
try {
    $sourceFingerprint = ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($fingerprintText)))).Replace("-", "").ToLowerInvariant()
} finally { $sha.Dispose() }

if ($WorkerOnly) {
    Invoke-Docker image inspect $ApiImage | Out-Null
    Write-Host "Reusing unchanged API image $ApiImage"
} else {
    Write-Host "Building $ApiImage for $TargetPlatform"
    Invoke-Docker buildx build --platform $TargetPlatform --load --provenance=false --sbom=false --progress plain `
        --build-arg "APP_VERSION=$ApiVersion" --build-arg "VCS_REF=$revision" `
        --file (Join-Path $ProjectRoot "docker\Dockerfile.api") --tag $ApiImage $ProjectRoot
}

Write-Host "Building $WorkerImage for $TargetPlatform"
Invoke-Docker buildx build --platform $TargetPlatform --load --provenance=false --sbom=false --progress plain `
    --build-arg "APP_VERSION=$WorkerVersion" --build-arg "VCS_REF=$revision" `
    --file (Join-Path $ProjectRoot "docker\Dockerfile.worker") --tag $WorkerImage $ProjectRoot

Invoke-Docker pull --platform $TargetPlatform $PostgresImage
Invoke-Docker pull --platform $TargetPlatform $RedisImage

Write-Host "Validating image contents"
Invoke-Docker run --rm --pull never --platform $TargetPlatform --entrypoint python $ApiImage -c `
    "import importlib.util; assert importlib.util.find_spec('torch') is None; assert importlib.util.find_spec('transformers') is None; print('API heavy OCR dependencies: absent')"
Invoke-Docker run --rm --pull never --platform $TargetPlatform --entrypoint sh $ApiImage -c `
    "test ! -e /app/tests && test ! -e /app/testdata && test ! -e /app/.git && test ! -e /models"
Invoke-Docker run --rm --pull never --platform $TargetPlatform --entrypoint python $WorkerImage -c `
    "import torch, transformers; print('worker torch=' + torch.__version__ + ' transformers=' + transformers.__version__)"
Invoke-Docker run --rm --pull never --platform $TargetPlatform --entrypoint sh $WorkerImage -c `
    "test ! -e /app/tests && test ! -e /app/testdata && test ! -e /app/.git && test ! -e /models/huggingface/hub/models--zenosai--MonkeyOCRv2-B-Parsing"

New-Item -ItemType Directory -Force -Path (Split-Path $ImageArchive) | Out-Null
if (Test-Path -LiteralPath $ImageArchive) { throw "Refusing to overwrite existing archive: $ImageArchive" }
Write-Host "Saving offline image archive"
Invoke-Docker save --output $ImageArchive $ApiImage $WorkerImage $PostgresImage $RedisImage
if ((Get-Item -LiteralPath $ImageArchive).Length -le 0) { throw "docker save produced an empty archive" }

$archiveHash = (Get-FileHash -LiteralPath $ImageArchive -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "Verifying archive with docker load"
Invoke-Docker load --input $ImageArchive
$expectedImages = @($ApiImage, $WorkerImage, $PostgresImage, $RedisImage)
foreach ($image in $expectedImages) { Invoke-Docker image inspect $image | Out-Null }

$imageEntries = foreach ($tag in $expectedImages) {
    $inspection = (& $Docker image inspect $tag | ConvertFrom-Json)[0]
    if ($LASTEXITCODE -ne 0) { throw "Could not inspect $tag" }
    $digest = if ($inspection.RepoDigests) { ($inspection.RepoDigests | Select-Object -First 1).Split("@")[-1] } else { $null }
    [ordered]@{ tag = $tag; id = $inspection.Id; digest = $digest; size_bytes = [int64]$inspection.Size }
}

$manifest = [ordered]@{
    schema_version = 1
    package_version = $Version
    build_status = "complete"
    build_date_utc = [DateTime]::UtcNow.ToString("o")
    git_commit = $gitCommit
    source_version = "pyproject:0.1.0; release:$Version"
    source_fingerprint_sha256 = $sourceFingerprint
    target_platform = $TargetPlatform
    images = $imageEntries
    postgres_version = "18.4-alpine"
    redis_version = "7.4.10-alpine"
    worker_base_image = "python:3.12-slim-bookworm"
    worker_cuda_runtime = "PyTorch 2.7.1 cu128 binary dependencies with Blackwell sm_120 support"
    worker_inference = "Verified MonkeyOCR local-safe path: SDPA, deterministic generation, KV cache, 1003520-pixel application limit"
    worker_count = 2
    worker_gpu_device_ids = @("0", "1")
    model = [ordered]@{
        included = $false
        expected_host_path = "/opt/models/MonkeyOCRv2-B-Parsing"
        container_path = "/models/huggingface"
        mount_mode = "read-only"
        model_id = "zenosai/MonkeyOCRv2-B-Parsing"
    }
    docker_images_archive = [ordered]@{
        path = "docker-images/$ImageArchiveName"
        sha256 = $archiveHash
        size_bytes = [int64](Get-Item -LiteralPath $ImageArchive).Length
        docker_load_verified = $true
    }
    required_server_components = @(
        "Linux x86_64", "Docker Engine 24+", "Docker Compose plugin 2+",
        "NVIDIA driver compatible with CUDA 12.8", "NVIDIA Container Toolkit",
        "curl", "sha256sum"
    )
}
$utf8 = New-Object System.Text.UTF8Encoding($false)
$manifestPath = Join-Path $DeploymentRoot "manifest.json"
[IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 10), $utf8)

$checksumTargets = @(
    $ImageArchive,
    (Join-Path $DeploymentRoot "VERSION"),
    (Join-Path $DeploymentRoot "API_IMAGE_VERSION"),
    (Join-Path $DeploymentRoot "WORKER_IMAGE_VERSION"),
    (Join-Path $DeploymentRoot "manifest.json"),
    (Join-Path $DeploymentRoot "compose.production.yml"),
    (Join-Path $DeploymentRoot "smoke\document.pdf"),
    (Join-Path $DeploymentRoot "smoke\reference.json")
) + (Get-ChildItem -LiteralPath (Join-Path $DeploymentRoot "scripts") -File | Select-Object -ExpandProperty FullName)
$checksumLines = foreach ($path in $checksumTargets) {
    $relative = $path.Substring($DeploymentRoot.Length + 1).Replace("\", "/")
    "$((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant())  $relative"
}
[IO.File]::WriteAllLines((Join-Path $DeploymentRoot "checksums.sha256"), $checksumLines, $utf8)

$cleanParent = Join-Path $env:TEMP ("spec-checker-deployment-clean-" + [guid]::NewGuid().ToString("N"))
$cleanDeployment = Join-Path $cleanParent "deployment"
New-Item -ItemType Directory -Force -Path $cleanParent | Out-Null
try {
    Copy-Item -LiteralPath $DeploymentRoot -Destination $cleanDeployment -Recurse
    Copy-Item -LiteralPath (Join-Path $cleanDeployment ".env.production.example") -Destination (Join-Path $cleanDeployment ".env.production")
    $config = & $Docker compose --env-file (Join-Path $cleanDeployment ".env.production") --file (Join-Path $cleanDeployment "compose.production.yml") config
    if ($LASTEXITCODE -ne 0) { throw "Compose config failed in clean directory" }
    $configText = $config -join "`n"
    if ($configText -match "(?m)^\s*build:") { throw "Clean Compose contains build:" }
    if ($configText -match "C:\\Users\\" -or $configText -match [regex]::Escape($ProjectRoot)) { throw "Clean Compose references a Windows/source path" }
    if ($configText -match "testdata|ground_truth") { throw "Clean Compose references test data" }
} finally {
    if (Test-Path -LiteralPath $cleanParent) { Remove-Item -LiteralPath $cleanParent -Recurse -Force }
}

if (Test-Path -LiteralPath $PackagePath) { throw "Refusing to overwrite existing package: $PackagePath" }
$pythonCandidates = @(
    (Join-Path $ProjectRoot ".venv-gpu\Scripts\python.exe"),
    (Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1)
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
if (-not $pythonCandidates) { throw "Python is required only on the build machine to create a portable tar.gz" }
& $pythonCandidates[0] (Join-Path $ProjectRoot "tools\create_deployment_archive.py") $DeploymentRoot $PackagePath
if ($LASTEXITCODE -ne 0) { throw "Could not create deployment tar.gz" }
$packageHash = (Get-FileHash -LiteralPath $PackagePath -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText("$PackagePath.sha256", "$packageHash  $PackageName`n", $utf8)

Write-Host "BUILD COMPLETE"
Write-Host "Images archive: $ImageArchive"
Write-Host "Images SHA-256: $archiveHash"
Write-Host "Deployment package: $PackagePath"
Write-Host "Package SHA-256: $packageHash"
