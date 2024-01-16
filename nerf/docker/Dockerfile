ARG BASE_IMAGE=ubuntu:22.04

FROM $BASE_IMAGE

ARG DEBIAN_FRONTEND=noninteractive

# Install base utilities
RUN apt-get update && \
    apt-get install -y apt-utils && \
    apt-get install -y libxml2 && \
    apt-get install -y git vim htop tmux wget && \
    apt-get install -y build-essential && \
    apt-get install -y manpages-dev && \
    apt-get install -y python3 && \
    apt-get install -y python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    alias python=python3 && \
    alias pip=pip3

RUN cd /root && wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run && \
    sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit --toolkitpath=/usr/local/cuda --override && rm -f cuda_11.8.0_520.61.05_linux.run

RUN printf "\nalias python=python3.8" >> /root/.bashrc && \
    printf "\nalias pip=pip3" >> /root/.bashrc

RUN pip3 install --upgrade pip && \
    pip3 install numpy torch torchvision ninja imageio PyOpenGL glfw xatlas gdown tensorboard configargparse && \
    pip3 install kornia plyfile scikit-image lpips wandb==0.15.2 gpustat omegaconf PyMCubes trimesh opencv-python

RUN pip3 install git+https://github.com/NVlabs/nvdiffrast/
    # pip3.8 install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch


ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/extras/CUPTI/lib64:/usr/local/lib:$LD_LIBRARY_PATH \
    PATH=$CUDA_PATH/bin:$PATH \
    CPLUS_INCLUDE_PATH=$CUDA_PATH/include:$CPLUS_INCLUDE_PATH

# Default pyopengl to EGL for good headless rendering support
ENV PYOPENGL_PLATFORM egl

COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN apt-get update && \
    apt-get install -y libglfw3-dev libgles2-mesa-dev && \
    apt-get install -y libglib2.0-0 rsync

COPY install.sh /root/install.sh

RUN apt-get install -y zsh curl && \
    sh -c /root/install.sh

COPY .zshrc /root/.zshrc

RUN  git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && \
    echo "source ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" >> ~/.zshrc && \
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    echo "source ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions/zsh-autosuggestions.zsh" >> ~/.zshrc && \
    chsh -s /bin/zsh

RUN echo "\nexport PATH="/usr/local/cuda/bin:$PATH"\nexport LIBRARY_PATH="/usr/local/cuda/lib64/stubs:$LIBRARY_PATH"" >> /root/.zshrc && \
    echo "alias python=python3" >> /root/.zshrc  && \
    echo "alias pip=pip3" >> /root/.zshrc

RUN pip3 install wandb mitsuba

RUN pip3 install nerfacc==0.5.0