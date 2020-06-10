| Branch | Status |
|:-------|:-------|
|master | [![pipeline status](https://code.ornl.gov/qci/tnqvm/badges/master/pipeline.svg)](https://code.ornl.gov/qci/tnqvm/commits/master) |


# TNQVM Tensor Network XACC Accelerator
These plugins for XACC provide an Accelerator implementation that leverages tensor network theory to simulate quantum circuits.

Installation
------------
With the XACC framework installed, run the following
```bash
$ mkdir build && cd build
$ cmake .. -DXACC_DIR=$HOME/.xacc (or wherever you installed XACC)
$ make install
```

TNQVM can be built with [ExaTN](https://github.com/ornl-qci/exatn) support, providing a tensor network processing backend that scales on Summit-like architectures. To enable this support, first follow the ExaTN [README](https://github.com/ORNL-QCI/exatn/blob/devel/README.md) to build and install ExaTN. Now configure TNQVM with CMake and build/install
```bash
$ mkdir build && cd build
$ cmake .. -DXACC_DIR=$HOME/.xacc -DEXATN_DIR=$HOME/.exatn
$ make install
```
To switch tensor processing backends use 
```
auto qpu = xacc::getAccelerator("tnqvm", {std::make_pair("tnqvm-visitor", "exatn")});
```
or in Python
```
qpu = xacc.getAccelerator('tnqvm', {'tnqvm-visitor':'exatn'})
```

MPI Execution
-------------
TNQVM's `exatn-mps` visitor can support multi-node execution via MPI. 

*Prerequisites*: ExaTN is built with MPI enabled, i.e., setting `MPI_LIB` and `MPI_ROOT_DIR` when configuring the ExaTN build.

To enable MPI in TNQVM, add `-DTNQVM_MPI_ENABLED=TRUE` to CMake along with other configuration variables.

A simulation executable which uses the `exatn-mps` visitor, e.g. via 
```
auto qpu = xacc::getAccelerator("tnqvm", { std::make_pair("tnqvm-visitor", "exatn-mps")});
```
can be executed with MPI using `mpiexec -np <number of processes> <executable>`.

Documentation
-------------

* [XACC Website and Documentation ](https://xacc.readthedocs.io)

Questions, Bug Reporting, and Issue Tracking
--------------------------------------------

Questions, bug reporting and issue tracking are provided by GitHub. Please
report all bugs by creating a new issue with the bug tag. You can ask
questions by creating a new issue with the question tag.

License
-------

TNQVM is licensed - [BSD 3-Clause](LICENSE).


Cite TNQVM
----------
If you use TNQVM in your research, please use the following citation
```
@article{tnqvm,
    author = {McCaskey, Alexander AND Dumitrescu, Eugene AND Chen, Mengsu AND Lyakh, Dmitry AND Humble, Travis},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {Validating quantum-classical programming models with tensor network simulations},
    year = {2018},
    month = {12},
    volume = {13},
    url = {https://doi.org/10.1371/journal.pone.0206704},
    pages = {1-19},
    number = {12},
    doi = {10.1371/journal.pone.0206704}
}
```
