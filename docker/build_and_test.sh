#!/bin/bash

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BLUE='\033[0;34m'
YELLOW='\033[1;33m'

# Function to print section headers
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

# Function to echo command before running
echo_run() {
    echo -e "${YELLOW}Running: $@${NC}"
    "$@"
}

# Function to build an image
build_image() {
    local type=$1
    print_header "Building $type image"
    echo_run docker build -t deepethogram:$type -f docker/Dockerfile-$type .
}

# Function to verify GPU in container
verify_gpu() {
    local gpu_flag=$1
    local type=$2
    echo "Verifying GPU access in container..."
    echo -e "${YELLOW}Running: docker run $gpu_flag --rm deepethogram:$type nvidia-smi${NC}"
    if ! docker run $gpu_flag --rm deepethogram:$type nvidia-smi; then
        echo -e "${RED}Failed to access GPU in container${NC}"
        return 1
    fi
    echo -e "${YELLOW}Running: docker run $gpu_flag --rm deepethogram:$type python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\"${NC}"
    if ! docker run $gpu_flag --rm deepethogram:$type python -c "import torch; print('CUDA available:', torch.cuda.is_available())" | grep -q "CUDA available: True"; then
        echo -e "${RED}Failed to access GPU through PyTorch${NC}"
        return 1
    fi
    return 0
}

# Function to run tests in container
test_container() {
    local type=$1
    local gpu_flag=$2
    local has_gpu=$3

    print_header "Testing $type container"

    # Test basic import
    echo "Testing Python import..."
    echo -e "${YELLOW}Running: docker run $gpu_flag -it deepethogram:$type python -c \"import deepethogram\"${NC}"
    docker run $gpu_flag -it deepethogram:$type python -c "import deepethogram" && \
        echo -e "${GREEN}✓ Import test passed${NC}" || \
        (echo -e "${RED}✗ Import test failed${NC}" && exit 1)

    # For containers that should support tests
    if [ "$type" = "full" ] || [ "$type" = "headless" ]; then
        echo "Running CPU tests..."
        echo -e "${YELLOW}Running: docker run $gpu_flag -it deepethogram:$type pytest -v -m \"not gpu\" tests/${NC}"
        docker run $gpu_flag -it deepethogram:$type pytest -v -m "not gpu" tests/ && \
            echo -e "${GREEN}✓ CPU tests passed${NC}" || \
            (echo -e "${RED}✗ CPU tests failed${NC}" && exit 1)

        # Run GPU tests if GPU is available
        if [ "$has_gpu" = true ] && [ "$type" != "gui" ]; then
            echo "Running GPU tests..."
            # First verify CUDA is accessible
            echo -e "${YELLOW}Running: docker run $gpu_flag -it deepethogram:$type python -c \"import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA is available')\"${NC}"
            docker run $gpu_flag -it deepethogram:$type python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA is available')"
            # Run the actual GPU tests
            echo -e "${YELLOW}Running: docker run $gpu_flag -it deepethogram:$type bash -c \"export CUDA_VISIBLE_DEVICES=0 && pytest -v -m gpu tests/\"${NC}"
            docker run $gpu_flag -it deepethogram:$type \
                bash -c "export CUDA_VISIBLE_DEVICES=0 && pytest -v -m gpu tests/" && \
                echo -e "${GREEN}✓ GPU tests passed${NC}" || \
                (echo -e "${RED}✗ GPU tests failed${NC}" && exit 1)
        fi
    fi

    # For containers that should support GUI
    if [ "$type" = "full" ] || [ "$type" = "gui" ]; then
        echo "Testing GUI import..."
        echo -e "${YELLOW}Running: docker run $gpu_flag -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw -it deepethogram:$type python -c \"from deepethogram.gui import main\"${NC}"
        docker run $gpu_flag -e DISPLAY=$DISPLAY \
            -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
            -it deepethogram:$type python -c "from deepethogram.gui import main" && \
            echo -e "${GREEN}✓ GUI import test passed${NC}" || \
            (echo -e "${RED}✗ GUI import test failed${NC}" && exit 1)
    fi
}

# Main execution
main() {
    # Ensure we're in the project root
    if [[ ! -f "pyproject.toml" ]]; then
        echo -e "${RED}Error: Must run from project root directory (where pyproject.toml is located)${NC}"
        exit 1
    fi

    # Check if GPU is available by testing nvidia-smi
    local has_gpu=false
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        GPU_FLAG="--gpus all"
        has_gpu=true
        echo -e "${GREEN}NVIDIA GPU detected, will use GPUs and run GPU tests${NC}"
    else
        GPU_FLAG=""
        echo -e "${RED}No NVIDIA GPU detected, running without GPU${NC}"
    fi

    # Build and test each image type
    for type in "headless" "gui" "full"; do
        build_image $type
        # Verify GPU access after building if we have a GPU
        if [ "$has_gpu" = true ] && [ "$type" != "gui" ]; then
            if ! verify_gpu "$GPU_FLAG" "$type"; then
                echo -e "${RED}GPU detected on host but not accessible in container. Please check nvidia-docker installation.${NC}"
                exit 1
            fi
        fi
        test_container $type "$GPU_FLAG" $has_gpu
    done

    print_header "All builds and tests completed successfully!"
}

# Execute main function
main
