# Use the Ubuntu base image
FROM ubuntu:latest

# Set the working directory
WORKDIR /app

# Install basic dependencies and add Debian testing repository
RUN apt-get update && \
    apt-get install -y software-properties-common wget gnupg2 dirmngr && \
    echo "deb http://deb.debian.org/debian testing main contrib non-free" > /etc/apt/sources.list.d/testing.list && \
    wget -O - https://ftp-master.debian.org/keys/archive-key-11.asc | apt-key add -

# Add Ubuntu Toolchain PPA and update package lists
RUN apt-get update && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update

# Install additional dependencies
RUN apt-get install -y \
    gcc-13 g++-13 libc6 libc6-dev libstdc++6 libopengl0 gmsh libgmsh-dev \
    libhdf5-dev libopenmpi-dev liblapack-dev python3.7 python3-pip && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 60 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 60 && \
    ln -s /usr/local/lib/libgmsh.so /usr/local/lib/libgmsh.so.4.12 && \
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Upgrade pip and install Python requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copy the application code
COPY . /app/

# Run the application
CMD ["./dist/nia_start_exe/nia_start_exe"]
