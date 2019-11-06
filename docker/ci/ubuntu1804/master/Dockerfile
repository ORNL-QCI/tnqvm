from xacc/ubuntu:18.04
run git clone --recursive https://github.com/eclipse/xacc && cd xacc && mkdir build && cd build \
    && cmake .. \
    && make -j$(nproc) install \
    && cd ../../ && git clone -b master https://github.com/ornl-qci/tnqvm && cd tnqvm && mkdir build && cd build \
    && cmake .. -DXACC_DIR=~/.xacc -DTNQVM_BUILD_TESTS=TRUE \
    && make -j$(nproc) install 