#!/bin/bash

export rpath_python=/opt/_internal/cpython-3.6.4/bin/python
git clone --recursive -b mccaskey/cpr_build https://github.com/ornl-qci/tnqvm
cd tnqvm

for version in cp36-cp36m cp35-cp35m cp34-cp34m cp33-cp33m cp27-cp27m cp27-cp27mu
do
        export myPython=/opt/python/$version/bin/python
        export ver=`case $version in "cp36-cp36m") echo 3.6 ;; "cp35-cp35m") echo 3.5 ;; "cp34-cp34m") echo 3.4 ;; "cp33-cp33m") echo 3.3 ;; "cp27-cp27m") echo 2.7 ;; "cp27-cp27mu") echo 2.7 ;; *) echo "invalid";; esac`
        PYTHONPATH=/xacc/build/lib.linux-x86_64-$ver/xacc $myPython setup.py build -t tmp_build --executable="/usr/bin/env python"
	PYTHONPATH=/xacc/build/lib.linux-x86_64-$ver/xacc $myPython setup.py bdist_wheel --skip-build
done
