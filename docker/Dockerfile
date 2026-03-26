FROM osrf/ros:humble-desktop

RUN apt update && apt install -y \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    colcon-common-extensions \
    carla==0.9.15 \
    numpy pandas matplotlib seaborn pygame-ce \
    networkx \
    opencv-python \
    msgpack \
    protobuf

# Auto source ROS
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

WORKDIR /workspace

CMD ["/bin/bash"]