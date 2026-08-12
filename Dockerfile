FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    git \
    wget \
    curl \
    aria2 \
    unzip \
    bc \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN python3.12 -m ensurepip --upgrade \
    && python3.12 -m pip install --no-cache-dir --upgrade pip \
    && ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip \
    && ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip3

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# ComfyUI v0.32.0 is the release that added LTX-2.5 support (PRs #15499, #15501,
# merged 2026-08-11, same day as the model). It ships:
#   - UNETLoader support for the LTX-2.5 transformer
#   - CLIPLoader(type="ltxv") for the Gemma-4 text encoder
#   - LTXVDualCFGGuider (replaces CFGGuider)
#   - the DiffVAE decoder (comfy/ldm/lightricks/vae/na_diffusion_decoder.py)
#   - int8_tensorwise + convrot dequant (comfy/ops.py:1170,1178)
# Pin EXACTLY. v0.21.0 (the LTX-2.3 service's pin) has none of the above.
# Do not float to a later tag without re-running the spike.
RUN git clone --branch v0.32.0 --depth 1 https://github.com/comfyanonymous/ComfyUI.git /comfyui
WORKDIR /comfyui
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir runpod huggingface_hub

# NOTE: diffusion_models (not checkpoints) — LTX-2.5 ships per-component files and
# is loaded with UNETLoader, which reads folder_paths' "diffusion_models" entry
# (folder_paths.py registers models/unet as a legacy alias for the same list).
RUN mkdir -p models/diffusion_models \
    models/text_encoders \
    models/vae \
    models/loras \
    models/latent_upscale_models \
    input \
    output \
    workflows

WORKDIR /comfyui/custom_nodes
# `|| true` is retained deliberately: several of these nodes' requirements.txt
# pin conflicting torch versions, and a hard failure here would block the build
# for a dependency we immediately override below. The safety net is
# check_node_registration.sh, which boots ComfyUI and fails the build if any
# class used by our workflows did not register. A silent import failure
# therefore becomes a NAMED BUILD FAILURE rather than a runtime surprise.
RUN git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git && \
    cd ComfyUI-LTXVideo && \
    git checkout "${LTXVIDEO_SHA:-master}" && \
    pip install --no-cache-dir -r requirements.txt || true

RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && \
    pip install --no-cache-dir -r requirements.txt || true

RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git && \
    cd ComfyUI-KJNodes && \
    pip install --no-cache-dir -r requirements.txt || true

RUN git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git && \
    cd ComfyUI-Custom-Scripts && \
    pip install --no-cache-dir -r requirements.txt || true

# Torch LAST, force-reinstalled: the custom-node requirements above downgrade it.
# This ordering is load-bearing — do not move it.
RUN pip install --no-cache-dir --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# v0.32.0's release notes state "Minimum officially supported pytorch is now 2.7".
# Fail the BUILD rather than the worker if the resolved cu128 wheel is older.
RUN python -c "import torch,sys; v=tuple(map(int,torch.__version__.split('+')[0].split('.')[:2])); assert v>=(2,7), f'torch {torch.__version__} < 2.7 required by ComfyUI v0.32.0'; print('torch', torch.__version__)"

COPY handler.py /handler.py
COPY start.sh /start.sh
COPY workflow_generated_audio.json /workflow_generated_audio.json
COPY workflow_custom_audio.json /workflow_custom_audio.json
COPY tools/check_node_map.py /tools/check_node_map.py
COPY tools/check_node_registration.sh /tools/check_node_registration.sh
RUN chmod +x /start.sh /tools/check_node_registration.sh

RUN ln -sf /workflow_generated_audio.json /workflow.json

# Item 7a — static: every mapped node ID exists, has the expected class_type,
# and the mapped input key is already present. Catches ID drift from the flatten.
RUN python /tools/check_node_map.py

# Item 7b — runtime: boot ComfyUI with NO models and assert every class_type used
# by either workflow is registered in /object_info. This is the only check that
# catches a custom-node import failure (see the `|| true` note above).
RUN /tools/check_node_registration.sh

WORKDIR /
EXPOSE 8188
CMD ["/start.sh"]
