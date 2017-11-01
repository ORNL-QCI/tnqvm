#ifndef TNQVM_ITENSOR_MPS_PROBNODE_HPP_
#define TNQVM_ITENSOR_MPS_PROBNODE_HPP_

#include <stdio.h>

namespace tnqvm {
class ProbNode{
public:
    ProbNode() :
        val (-99.),
        iqbit (-1), // must be -1, because its child is the probability for qbit 0
        left (NULL),
        right (NULL), outcome(-1) {}

    ProbNode(double val_in, int qbitIdx, int otcome) :
        val (val_in),
        left (NULL),
        right (NULL), outcome(otcome), iqbit(qbitIdx) {}
    double val;
    int iqbit;
    int outcome;
    ProbNode* left;
    ProbNode* right;

    void print(){
        std::cout<<"("<<iqbit<<", "<<outcome<<", "<<val<<")"<<std::endl;
    }

    ~ProbNode(){
        if (left!=NULL){
            delete left;
        }
        if (right!=NULL){
            delete right;
        }
    }
};
}

#endif /* TNQVM_ITENSOR_MPS_PROBNODE_HPP_ */
