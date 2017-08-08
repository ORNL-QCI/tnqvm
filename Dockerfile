# change to base linux image laster
from mchen360/yaccdev  
RUN cd /allacc && \
    source ~/.bashrc && \
    git clone --recursive https://github.com/mileschen360/tnqvm.git && \
    cd tnqvm && mkdir build && cd build && \
    cmake .. && \
    make install