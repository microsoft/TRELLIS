FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y ffmpeg build-essential git rdfind

WORKDIR /app
ADD setup.sh   /app/setup.sh

# Setup conda
RUN conda config --set always_yes true && \
    conda init && \
    conda config --add channels defaults

# Use bash shell so we can source activate
SHELL ["/bin/bash", "-c"]


RUN conda install torchvision=0.19.0 \
                  onnx==1.17.0 \
                  pytorch=2.4.0 \
                  pytorch-cuda=12.4 \
              --force-reinstall \
              -c pytorch \
              -c nvidia

# Create a g++ wrapper for JIT, since the include dirs are passed with -i rather than -I for some reason
RUN printf '#!/usr/bin/env bash\nexec /usr/bin/g++ -I/usr/local/cuda/include -I/usr/local/cuda/include/crt "$@"\n' > /usr/local/bin/gxx-wrapper && \
    chmod +x /usr/local/bin/gxx-wrapper
ENV CXX=/usr/local/bin/gxx-wrapper


# Run setup.sh - this won't install all the things due to missing GPU in the builder
RUN conda run -n base ./setup.sh --basic --xformers --flash-attn --diffoctreerast --vox2seq --spconv --mipgaussian --kaolin --nvdiffrast --demo

# Now install additional Python packages
# These ones work inside the builder
RUN conda run -n base pip install diso
RUN conda run -n base pip install plyfile utils3d flash_attn spconv-cu120 xformers onnxscript
RUN conda run -n base pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
RUN conda run -n base pip install git+https://github.com/NVlabs/nvdiffrast.git

# Remove downloaded packages from conda and pip
RUN conda clean --all -f -y
RUN pip cache purge

# Deduplicate with rdfind
# This reduces the size of the image by a few hundred megs.
RUN rdfind -makesymlinks true /opt/conda

# Final stage
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel AS final

WORKDIR /app
COPY --from=builder /usr/local/bin/gxx-wrapper /usr/local/bin/gxx-wrapper
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /root /root
COPY --from=builder /app /app

# Reinstall any runtime tools needed
# git and build-essential are needed for post_install.sh script.
# vim and strace are useful for debugging, remove those if you want to.
RUN apt update && \
    apt upgrade -y && \
    apt install -y build-essential \
                       git \
                       strace \
                       vim && \
    rm -rf /var/lib/apt/lists/*

# install these last, so we can experiment without excessive build times.
COPY trellis         /app/trellis
COPY app.py          /app/app.py
COPY example.py      /app/example.py
COPY extensions      /app/extensions
COPY assets          /app/assets
COPY onstart.sh      /app/onstart.sh
COPY post_install.sh /app/post_install.sh

ENV PATH=/opt/conda/bin:$PATH

# This script runs the post_install steps

# If you're pushing to a container registry, let this run once, run some
# tests, then do `docker commit` to save the models along with the image.
# This will ensure that it won't fail at runtime due to models being
# unavailable, or network restrictions.
CMD ["bash", "/app/onstart.sh"]

