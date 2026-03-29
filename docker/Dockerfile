ARG UBUNTU_VERSION=22.04
ARG CPU_BASE=ubuntu:${UBUNTU_VERSION}
ARG CUDA_BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ARG CUDA_PLATFORM=linux/amd64
ARG UV_VERSION=0.6.14

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin
ARG APP_USER=appuser
ARG APP_UID=1000
ARG APP_GID=1000

FROM ${CPU_BASE} AS os-base-cpu
ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libx11-6 \
    libxext6 \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# CUDA targets stay amd64-only because the NVIDIA base images used here do not
# provide native arm64 variants for this workflow.
FROM --platform=${CUDA_PLATFORM} ${CUDA_BASE} AS os-base-cuda
ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libx11-6 \
    libxext6 \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

FROM os-base-cpu AS os-base-gui
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libasound2 \
    libdbus-1-3 \
    libdouble-conversion3 \
    libegl1 \
    libfontconfig1 \
    libgl1 \
    libnss3 \
    libopengl0 \
    libpulse-mainloop-glib0 \
    libx11-xcb1 \
    libxcb-cursor0 \
    libxcb-glx0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-util1 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

FROM os-base-cuda AS os-base-full
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libasound2 \
    libdbus-1-3 \
    libdouble-conversion3 \
    libegl1 \
    libfontconfig1 \
    libgl1 \
    libnss3 \
    libopengl0 \
    libpulse-mainloop-glib0 \
    libx11-xcb1 \
    libxcb-cursor0 \
    libxcb-glx0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-sync1 \
    libxcb-util1 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

FROM os-base-gui AS builder-gui
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH
WORKDIR /app
COPY pyproject.toml uv.lock README.md MANIFEST.in ./
COPY --from=uv-bin /uv /uvx /bin/
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --python /usr/bin/python3 --no-dev --no-install-project
COPY deepethogram ./deepethogram
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --python /usr/bin/python3 --no-dev --no-editable

FROM os-base-cuda AS builder-headless
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH
WORKDIR /app
COPY pyproject.toml uv.lock README.md MANIFEST.in ./
COPY --from=uv-bin /uv /uvx /bin/
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --python /usr/bin/python3 --no-dev --no-install-project
COPY deepethogram ./deepethogram
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --python /usr/bin/python3 --no-dev --no-editable

FROM os-base-full AS builder-full
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH
WORKDIR /app
COPY pyproject.toml uv.lock README.md MANIFEST.in ./
COPY --from=uv-bin /uv /uvx /bin/
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --python /usr/bin/python3 --no-dev --no-install-project
COPY deepethogram ./deepethogram
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --python /usr/bin/python3 --no-dev --no-editable

FROM os-base-gui AS gui
ARG APP_USER=appuser
ARG APP_UID=1000
ARG APP_GID=1000
ENV DEG_VERSION=gui \
    PATH=/opt/venv/bin:$PATH
WORKDIR /workspace
COPY --from=builder-gui /opt/venv /opt/venv
RUN groupadd --gid "${APP_GID}" "${APP_USER}" || true \
    && useradd --create-home --home-dir "/home/${APP_USER}" --uid "${APP_UID}" --gid "${APP_GID}" "${APP_USER}" \
    && install -d -o "${APP_UID}" -g "${APP_GID}" /workspace
USER ${APP_USER}
CMD ["python", "-m", "deepethogram"]

FROM os-base-cuda AS headless
ARG APP_USER=appuser
ARG APP_UID=1000
ARG APP_GID=1000
ENV DEG_VERSION=headless \
    PATH=/opt/venv/bin:$PATH
WORKDIR /workspace
COPY --from=builder-headless /opt/venv /opt/venv
RUN groupadd --gid "${APP_GID}" "${APP_USER}" || true \
    && useradd --create-home --home-dir "/home/${APP_USER}" --uid "${APP_UID}" --gid "${APP_GID}" "${APP_USER}" \
    && install -d -o "${APP_UID}" -g "${APP_GID}" /workspace
USER ${APP_USER}
CMD ["python", "-m", "deepethogram"]

FROM os-base-full AS full
ARG APP_USER=appuser
ARG APP_UID=1000
ARG APP_GID=1000
ENV DEG_VERSION=full \
    PATH=/opt/venv/bin:$PATH
WORKDIR /workspace
COPY --from=builder-full /opt/venv /opt/venv
RUN groupadd --gid "${APP_GID}" "${APP_USER}" || true \
    && useradd --create-home --home-dir "/home/${APP_USER}" --uid "${APP_UID}" --gid "${APP_GID}" "${APP_USER}" \
    && install -d -o "${APP_UID}" -g "${APP_GID}" /workspace
USER ${APP_USER}
CMD ["python", "-m", "deepethogram"]

FROM builder-gui AS test-gui
ENV DEG_VERSION=gui
COPY pytest.ini setup_tests.py ./
COPY tests ./tests
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    openbox \
    xauth \
    xvfb \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --no-editable
CMD ["sh", "-lc", "python setup_tests.py && xvfb-run --auto-servernum --server-args='-screen 0 1920x1200x24' sh -lc 'openbox >/tmp/openbox.log 2>&1 & pytest -v -m \"not gpu\"'"]

FROM builder-headless AS test-headless
COPY pytest.ini setup_tests.py ./
COPY tests ./tests
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --no-editable
CMD ["pytest", "-v"]

FROM builder-full AS test-full
COPY pytest.ini setup_tests.py ./
COPY tests ./tests
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --locked --no-editable
CMD ["pytest", "-v"]
