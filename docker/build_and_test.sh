#!/usr/bin/env bash

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'

print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

echo_run() {
    echo -e "${YELLOW}Running: $*${NC}"
    "$@"
}

build_image() {
    local target=$1
    local platform=${2:-}
    local args=(docker buildx build --load --target "$target" -t "deepethogram:$target" -f docker/Dockerfile)

    if [[ -n "$platform" ]]; then
        args+=(--platform "$platform")
    fi

    args+=(.)

    print_header "Building $target image"
    echo_run "${args[@]}"
}

verify_gpu() {
    local image=$1
    local gpu_flag=$2

    print_header "Verifying GPU access for ${image}"
    echo_run docker run ${gpu_flag:+$gpu_flag} --rm "$image" nvidia-smi
    docker run ${gpu_flag:+$gpu_flag} --rm "$image" nvidia-smi >/dev/null

    echo_run docker run ${gpu_flag:+$gpu_flag} --rm "$image" python -c "import torch; assert torch.cuda.is_available(); print('CUDA available')"
    docker run ${gpu_flag:+$gpu_flag} --rm "$image" python -c "import torch; assert torch.cuda.is_available(); print('CUDA available')" >/dev/null
}

smoke_test_runtime() {
    local target=$1
    local gpu_flag=$2

    print_header "Smoke testing $target runtime image"
    echo_run docker run ${gpu_flag:+$gpu_flag} --rm "deepethogram:$target" python -c "import deepethogram"
    docker run ${gpu_flag:+$gpu_flag} --rm "deepethogram:$target" python -c "import deepethogram"

    if [[ "$target" == "gui" || "$target" == "full" ]]; then
        echo_run docker run ${gpu_flag:+$gpu_flag} --rm "deepethogram:$target" python -c "from deepethogram.gui import main"
        docker run ${gpu_flag:+$gpu_flag} --rm "deepethogram:$target" python -c "from deepethogram.gui import main"
    fi
}

run_pytest_target() {
    local target=$1
    local gpu_flag=$2
    local marker=$3
    local image="deepethogram:$target"
    local mount_args=()

    if [[ -d tests/DATA ]]; then
        mount_args=(-v "$PWD/tests/DATA:/app/tests/DATA:ro")
    else
        echo -e "${YELLOW}Skipping pytest for ${target}: tests/DATA is not present locally.${NC}"
        return 0
    fi

    print_header "Running ${marker} tests in ${target}"
    echo_run docker run ${gpu_flag:+$gpu_flag} --rm "${mount_args[@]}" "$image" pytest -v -m "$marker" tests/
    docker run ${gpu_flag:+$gpu_flag} --rm "${mount_args[@]}" "$image" pytest -v -m "$marker" tests/
}

main() {
    if [[ ! -f pyproject.toml ]]; then
        echo -e "${RED}Error: run this script from the repository root.${NC}"
        exit 1
    fi

    if ! docker buildx version >/dev/null 2>&1; then
        echo -e "${RED}Error: docker buildx is required.${NC}"
        exit 1
    fi

    local has_gpu=false
    local gpu_flag=""
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        has_gpu=true
        gpu_flag="--gpus all"
        echo -e "${GREEN}NVIDIA GPU detected; CUDA runtime checks are enabled.${NC}"
    else
        echo -e "${YELLOW}No NVIDIA GPU detected; CUDA runtime checks will be skipped.${NC}"
    fi

    build_image gui
    smoke_test_runtime gui ""

    build_image headless linux/amd64
    smoke_test_runtime headless "$gpu_flag"
    build_image test-headless linux/amd64
    run_pytest_target test-headless "$gpu_flag" "not gpu"

    build_image full linux/amd64
    smoke_test_runtime full "$gpu_flag"
    build_image test-full linux/amd64

    if [[ "$has_gpu" == true ]]; then
        verify_gpu deepethogram:headless "$gpu_flag"
        verify_gpu deepethogram:full "$gpu_flag"
        run_pytest_target test-full "$gpu_flag" "gpu"
    else
        echo -e "${YELLOW}Skipping GPU verification and GPU-marked tests because no NVIDIA GPU is available.${NC}"
    fi

    print_header "All requested Docker builds completed successfully"
}

main "$@"
