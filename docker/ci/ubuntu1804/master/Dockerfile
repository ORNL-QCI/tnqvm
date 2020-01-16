from xacc/ubuntu:18.04
run git clone --recursive https://github.com/eclipse/xacc && cd xacc && mkdir build && cd build \
    && CC=gcc-8 CXX=g++-8 cmake .. \
    && make -j$(nproc) install \
    && cd ../../ && git clone --recursive https://github.com/ornl-qci/exatn && cd exatn && mkdir build && cd build \
    && CC=gcc-8 CXX=g++-8 FC=gfortran-8 cmake .. -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
                -DBLAS_LIB=ATLAS \
                -DBLAS_PATH=/usr/lib/x86_64-linux-gnu && make -j$(nproc) install && cd ../.. \
    && git clone -b master https://github.com/ornl-qci/tnqvm && cd tnqvm && mkdir build && cd build \
    && CC=gcc-8 CXX=g++-8 GC=gfortran-8 cmake .. -DXACC_DIR=~/.xacc -DTNQVM_BUILD_TESTS=TRUE -DEXATN_DIR=~/.exatn \
    && make -j$(nproc) install