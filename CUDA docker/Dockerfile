FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

########################################################
# Essential packages for remote debugging and login in
########################################################

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    apt-utils gcc g++ openssh-server cmake build-essential gdb gdbserver rsync vim wget git

# Install CMake 3.16
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.20.1/cmake-3.20.1-Linux-x86_64.sh && \
    chmod +x cmake-3.20.1-Linux-x86_64.sh && \
    ./cmake-3.20.1-Linux-x86_64.sh --skip-license --prefix=/usr/local && \
    rm cmake-3.20.1-Linux-x86_64.sh

# Copy CUDA Toolkit installer from local directory
COPY cuda_12.4.1_550.54.15_linux.run /tmp/cuda_12.4.1_550.54.15_linux.run

# Install CUDA Toolkit from local installer
RUN chmod +x /tmp/cuda_12.4.1_550.54.15_linux.run && \
    /tmp/cuda_12.4.1_550.54.15_linux.run --toolkit --silent && \
    rm /tmp/cuda_12.4.1_550.54.15_linux.run

RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# 22 for ssh server. 7777 for gdb server.
EXPOSE 22 7777

RUN useradd -ms /bin/bash debugger
RUN echo 'debugger:pwd' | chpasswd

########################################################
# Add custom packages and development environment here
########################################################
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.4/lib64
ENV PATH=$PATH:/usr/local/cuda-12.4/bin
ENV CUDA_HOME=/usr/local/cuda-12.4
########################################################

CMD ["/usr/sbin/sshd", "-D"]
