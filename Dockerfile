FROM ubuntu:latest

# Install necessary utilities and libraries
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    libudev1 \
    libsystemd0 \
    libc6 \
    software-properties-common \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libxrandr2 \
    libxcursor1 \
    libfreetype6 \
    libfontconfig1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libgl1 \
    libglu1-mesa \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    mpich \
    libmpich-dev

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda clean --all --yes && \
    ln -s /opt/conda/bin/conda /usr/bin/conda

# Add Miniconda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Install GCC 13
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y && \
    apt-get update && \
    apt-get install -y gcc-13 g++-13 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 1 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 1

# Copy the contents of the nia_start_exe directory into the container
COPY dist/nia_start_exe /app/nia_start_exe
COPY meshes/ /app/nia_start_exe
COPY configs/ /app/nia_start_exe
WORKDIR /app/nia_start_exe

# Copy the requirements.txt file
COPY requirements.txt /app/nia_start_exe/requirements.txt

# Create and activate the Conda environment, install dependencies
RUN conda create -n startenv python=3.7 --yes && \
    echo "source activate startenv" > ~/.bashrc && \
    conda run -n startenv pip install --upgrade pip && \
    conda run -n startenv pip install -r requirements.txt

# Set environment variables for Qt5
ENV LD_LIBRARY_PATH=/opt/conda/envs/startenv/lib/python3.7/site-packages/PyQt5/Qt5/lib:$LD_LIBRARY_PATH

# Set the entry point to the executable file
CMD ["./nia_start_exe"]
