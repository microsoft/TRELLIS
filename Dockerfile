FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg build-essential htop git python3-onnx rdfind

WORKDIR /app

# Add the application files
ADD trellis    /app/trellis
ADD setup.sh   /app/setup.sh
ADD headless_app.py /app/headless_app.py
ADD example.py /app/example.py
ADD extensions /app/extensions
ADD assets     /app/assets

# Initialize and update git submodules
RUN cd /app && \
    git init && \
    git submodule init && \
    git submodule update --init --recursive

# Setup conda
RUN conda config --set always_yes true && conda init

# Use bash shell so we can source activate
SHELL ["/bin/bash", "-c"]

RUN conda install cuda=12.4 pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# Create a g++ wrapper for JIT, since the include dirs are passed with -i rather than -I for some reason
RUN printf '#!/usr/bin/env bash\nexec /usr/bin/g++ -I/usr/local/cuda/include -I/usr/local/cuda/include/crt "$@"\n' > /usr/local/bin/gxx-wrapper && \
    chmod +x /usr/local/bin/gxx-wrapper
ENV CXX=/usr/local/bin/gxx-wrapper

# Run setup.sh - this won't install all the things, we'll need to install some later
RUN conda run -n base ./setup.sh --basic --xformers --flash-attn --diffoctreerast --vox2seq --spconv --mipgaussian --kaolin --nvdiffrast --demo

# Now install additional Python packages
# These ones work inside the builder
RUN conda run -n base pip install diso
RUN conda run -n base pip install plyfile utils3d flash_attn spconv-cu120 xformers
RUN conda run -n base pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
RUN conda run -n base pip install git+https://github.com/NVlabs/nvdiffrast.git

# Cleanup after builds are done
RUN apt-get remove -y ffmpeg build-essential htop git python3-onnx && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN conda clean --all -f -y

# Deduplicate with rdfind
# This reduces the size of the image by a few hundred megs. Not great, but it's a start.
RUN rdfind -makesymlinks true /opt/conda

# Final stage
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel AS final

WORKDIR /app
COPY --from=builder /usr/local/bin/gxx-wrapper /usr/local/bin/gxx-wrapper
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /root /root
COPY --from=builder /app /app

# Reinstall any runtime tools needed
# git and build-essential are needed for post_install.sh script. vim and strace are
# useful for debugging the image size.
RUN apt-get update && \
    apt-get install -y build-essential \
                       git \
                       strace \
                       vim && \
    rm -rf /var/lib/apt/lists/*

# Add FastAPI dependencies
RUN conda run -n base pip install fastapi uvicorn python-multipart

# Add the new startup script
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

ENV PATH=/opt/conda/bin:$PATH

# This script runs the post_install steps
CMD ["/app/startup.sh"]