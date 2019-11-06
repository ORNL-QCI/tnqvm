from xacc/ubuntu:18.04
run git clone --recursive -b xacc-devel https://github.com/eclipse/xacc && cd xacc && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) install \
    && cd ../../ && git clone -b devel https://github.com/ornl-qci/tnqvm && cd tnqvm && mkdir build && cd build \
    && cmake .. -DXACC_DIR=~/.xacc \
    && make -j$(nproc) install
