


#ifndef TNQVM_TNQVMBUFFER_HPP_
#define TNQVM_TNQVMBUFFER_HPP_

#include "xacc/AcceleratorBuffer.hpp"


class TNQVMBuffer : public xacc::AcceleratorBuffer{
public:

    TNQVMBuffer(const std::string& str, const int N)
        : xacc::AcceleratorBuffer(str,N),
        aver_from_wavefunc (1.),
        aver_from_manytime (1.) {}

    void resetBuffer(){
        xacc::AcceleratorBuffer::resetBuffer();
        aver_from_wavefunc = 1.;
        aver_from_manytime = 1.;
    }

    double aver_from_wavefunc;
    double aver_from_manytime;
};
#endif