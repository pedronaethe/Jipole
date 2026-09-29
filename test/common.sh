#defining directories for ipole too for tests that require ipole
TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TEST_ROOT/.." && pwd)"
CACHE_DIR="${JIPOLE_TEST_CACHE:-$TEST_ROOT/.cache}"
IPOLE_DIR="$TEST_ROOT/ipole"
IPOLE_REPO="https://github.com/AFD-Illinois/ipole.git"
IPOLE_REF="7f7a482"
JULIA_VERSION="$(grep -m1 '^julia_version' "$REPO_ROOT/scripts/Manifest.toml" | cut -d'"' -f2)"
JULIA_CHANNEL="${JULIA_VERSION%.*}"
NPROC="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"

mkdir -p "$CACHE_DIR"


#Here I'll also define functions used in the run.sh files.

log() {
    echo "[$(basename "$PWD")] $*" >&2
}

apt_install() {
    command -v apt-get > /dev/null || return 1
    local sudo=""
    if [[ "$(id -u)" -ne 0 ]]; then
        sudo -n true 2> /dev/null || return 1
        sudo="sudo -n"
    fi
    log "Installing $*"
    $sudo apt-get update -qq > /dev/null &&
        DEBIAN_FRONTEND=noninteractive $sudo apt-get install -y -qq "$@" > /dev/null
}

has_c_header() {
    echo "#include <$1>" | gcc -E -x c - > /dev/null 2>&1
}

hdf5_make_args() {
    if [[ -n "${IPOLE_CC:-}" ]]; then
        MAKE_ARGS=(CC="$IPOLE_CC")
        return 0
    fi
    local wrapper
    for wrapper in h5cc h5pcc; do
        if command -v "$wrapper" > /dev/null; then
            MAKE_ARGS=(CC="$wrapper -shlib")
            return 0
        fi
    done
    local cflags libdir
    if [[ -n "${HDF5_PATH:-}" ]]; then
        cflags="-I$HDF5_PATH/include"
        libdir="-L$HDF5_PATH/lib -Wl,-rpath,$HDF5_PATH/lib"
    elif pkg-config --exists hdf5 2> /dev/null; then
        cflags="$(pkg-config --cflags hdf5)"
        libdir="$(pkg-config --libs-only-L hdf5)"
    else
        return 1
    fi
    MAKE_ARGS=(CC="gcc $cflags" LIBDIR="$libdir" LIB="-lm -lgsl -lgslcblas -lhdf5_hl -lhdf5")
}

ensure_system_deps() {
    local packages=() tool
    for tool in gcc make git curl; do
        command -v "$tool" > /dev/null || packages+=("$tool")
    done
    has_c_header gsl/gsl_math.h || packages+=(libgsl-dev)
    hdf5_make_args || packages+=(libhdf5-dev hdf5-helpers pkg-config)
    [[ ${#packages[@]} -eq 0 ]] && return 0
    apt_install "${packages[@]}" || {
        log "Missing and could not install: ${packages[*]}"
        return 1
    }
    hdf5_make_args
}


# This function will make sure ipole exists already. Multiple tests might have to clone ipole in the future, so good to have this here.
#if it doesn't exist, just clone it and compile.
#make sure ipole is compiled with the right model...
ensure_ipole() {
    local model="$1"
    IPOLE_BIN="$IPOLE_DIR/ipole_$model"
    [[ -x "$IPOLE_BIN" ]] && return 0
    ensure_system_deps || return 1
    if [[ ! -d "$IPOLE_DIR/.git" ]]; then
        log "Cloning ipole $IPOLE_REF"
        rm -rf "$IPOLE_DIR"
        git clone -q "$IPOLE_REPO" "$IPOLE_DIR" || return 1
    fi
    git -C "$IPOLE_DIR" checkout -q "$IPOLE_REF" || return 1
    log "Building ipole (MODEL=$model)"
    make -C "$IPOLE_DIR" clean > /dev/null 2>&1
    if ! make -C "$IPOLE_DIR" -j "$NPROC" MODEL="$model" "${MAKE_ARGS[@]}" > "$IPOLE_DIR/build_$model.log" 2>&1; then
        tail -30 "$IPOLE_DIR/build_$model.log" >&2
        log "ipole build failed"
        return 1
    fi
    cp "$IPOLE_DIR/ipole" "$IPOLE_BIN"
}

#ensure python makes sure python exists with all the necessary libraries...
ensure_python() {
    local py="${PYTHON:-python3}"
    if "$py" -c "import h5py, numpy, matplotlib" 2> /dev/null; then
        PYTHON="$py"
        return 0
    fi
    local venv="$CACHE_DIR/pyenv"
    if [[ ! -x "$venv/bin/python" ]]; then
        command -v python3 > /dev/null || apt_install python3 python3-venv || {
            log "python3 not found"
            return 1
        }
        python3 -m venv "$venv" 2> /dev/null || { apt_install python3-venv && python3 -m venv "$venv"; } || {
            log "Could not create a Python virtual environment"
            return 1
        }
    fi
    if ! "$venv/bin/python" -c "import h5py, numpy, matplotlib" 2> /dev/null; then
        log "Installing h5py, numpy, matplotlib"
        "$venv/bin/python" -m pip install -q --upgrade pip h5py numpy matplotlib || return 1
    fi
    PYTHON="$venv/bin/python"
}

julia_channel_of() {
    $1 -e 'print(VERSION.major, ".", VERSION.minor)' 2> /dev/null
}


#make sure julia is installed and in the right version...
ensure_julia() {
    if [[ -z "${JULIA:-}" ]]; then
        if ! command -v julia > /dev/null; then
            if [[ ! -x "$HOME/.juliaup/bin/julia" ]]; then
                log "Installing Julia $JULIA_CHANNEL"
                curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel "$JULIA_CHANNEL" > /dev/null || return 1
            fi
            export PATH="$HOME/.juliaup/bin:$PATH"
        fi
        JULIA="julia"
        if [[ "$(julia_channel_of "$JULIA")" != "$JULIA_CHANNEL" ]]; then
            if command -v juliaup > /dev/null; then
                juliaup status 2> /dev/null | grep -qE "^\s*\*?\s*$JULIA_CHANNEL\s" || {
                    log "Adding Julia $JULIA_CHANNEL with juliaup"
                    juliaup add "$JULIA_CHANNEL" > /dev/null 2>&1 || return 1
                }
                JULIA="julia +$JULIA_CHANNEL"
            fi
        fi
    fi
    if [[ "$(julia_channel_of "$JULIA")" != "$JULIA_CHANNEL" ]]; then
        log "Julia $JULIA_CHANNEL is needed (scripts/Manifest.toml), found $(julia_channel_of "$JULIA"); set JULIA to a Julia $JULIA_CHANNEL command"
        return 1
    fi
    if ! $JULIA --project="$REPO_ROOT/scripts" -e 'using Pkg; Pkg.instantiate()' > "$CACHE_DIR/julia_instantiate.log" 2>&1; then
        tail -30 "$CACHE_DIR/julia_instantiate.log" >&2
        log "Could not instantiate the Julia environment"
        return 1
    fi
}

run_jipole() {
    local par="$1"
    local log_file="output/$(basename "${par%.toml}").log"
    $JULIA --project="$REPO_ROOT/scripts" --threads="$NPROC" "$REPO_ROOT/scripts/generate_image.jl" "$par" > "$log_file" 2>&1 || {
        tail -20 "$log_file" >&2
        log "Jipole failed on $par"
        return 1
    }
}

file_md5() {
    if command -v md5sum > /dev/null; then
        md5sum "$1" | cut -d' ' -f1
    else
        md5 -q "$1"
    fi
}

ensure_file() {
    local name="$1" url="$2" md5="$3"
    local path="$CACHE_DIR/$name"
    if [[ ! -f "$path" ]]; then
        log "Downloading $name"
        curl -fL --retry 3 -s -S -o "$path.part" "$url" || return 1
        mv "$path.part" "$path"
    fi
    if [[ "$(file_md5 "$path")" != "$md5" ]]; then
        log "Checksum mismatch for $path, delete it to download it again"
        return 1
    fi
    echo "$path"
}
