# Installs jl on Windows: irm https://jarvislabs.ai/install.ps1 | iex
#
# Installs uv if missing, installs jl with `uv tool install`, puts it on PATH,
# then runs `jl setup` and `jl connect`.
#
# Environment:
#   JL_INSTALL_SPEC  what to install (default: jarvislabs); CI passes a wheel path.
#   JL_SKIP_SETUP    set to skip `jl setup` and `jl connect`.

# Errors throw instead of calling exit: under `irm | iex`, exit would close the user's window.

# What to run later, printed last so it's what the window ends on.
function Show-JlNext {
    Write-Host 'jl is installed. Next:'
    Write-Host '  jl status    see your coding tools'
    Write-Host '  jl connect   connect another one'
}

function Install-Jl {
    $ErrorActionPreference = 'Stop'
    # The PATH this window started with, before this script adds uv's and jl's folders to it.
    $startPath = $env:Path
    $spec = if ($env:JL_INSTALL_SPEC) { $env:JL_INSTALL_SPEC } else { 'jarvislabs' }

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host 'Installing uv...'
        # A uv release jl tested, from Astral's version-pinned installer; it moves with jl releases.
        # A child shell, so an exit in uv's installer can't close this one.
        powershell -ExecutionPolicy ByPass -NoProfile -Command 'irm https://astral.sh/uv/0.13.6/install.ps1 | iex'
        if ($LASTEXITCODE) { throw 'Installing uv failed.' }
        # Where uv's installer puts uv; it adds the folder to the user's PATH, not this session's.
        $uvBin = if ($env:UV_INSTALL_DIR) { $env:UV_INSTALL_DIR }
                 elseif ($env:XDG_BIN_HOME) { $env:XDG_BIN_HOME }
                 else { Join-Path $HOME '.local\bin' }
        $env:Path = "$uvBin;$env:Path"
    }

    Write-Host 'Installing jl...'
    uv tool install --upgrade $spec
    if ($LASTEXITCODE) { throw 'Installing jl failed.' }
    $bin = (Resolve-Path (uv tool dir --bin)).Path
    uv tool update-shell
    if ($LASTEXITCODE) { Write-Host "Add $bin to your PATH." }
    # irm | iex runs in the user's session, so jl also runs in this window, unlike after install.sh.
    $env:Path = "$bin;$env:Path"
    # An older jl earlier on the PATH would run in a new terminal instead of the one just installed.
    $after = $env:Path
    $env:Path = $startPath
    $earlier = Get-Command jl -ErrorAction SilentlyContinue | Select-Object -First 1
    $env:Path = $after
    if ($earlier.Source -and $earlier.Source -ne (Join-Path $bin 'jl.exe')) {
        Write-Host "Another jl at $($earlier.Source) comes first on your PATH: remove it, or open a new terminal."
    }

    if ($env:JL_SKIP_SETUP) { Show-JlNext; return }
    # Marks jl setup's run so it hands off to jl connect instead of printing its guide.
    # irm | iex runs in the user's session, so the marker comes off again however jl setup ends.
    $env:JL_FROM_INSTALLER = '1'
    try { jl setup } finally { Remove-Item Env:JL_FROM_INSTALLER -ErrorAction SilentlyContinue }
    # 130: Ctrl-C at one of jl's questions. The user stopped it, so it ends without an error.
    if ($LASTEXITCODE -eq 130) { return }
    if ($LASTEXITCODE) { throw '`jl setup` failed. Run it again, then `jl connect`.' }
    jl connect
    Show-JlNext
}

Install-Jl
