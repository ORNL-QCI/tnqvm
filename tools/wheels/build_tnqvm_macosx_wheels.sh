#!/bin/bash

mkdir -p $HOME/tnqvm-wheelhouse
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

for version in 3.6.4 #2.7.14-ucs2 2.7.14-ucs4 3.3.7 3.4.7 3.5.4 3.6.4
do
	pyenv virtualenv $version xacc-$version
	pyenv activate xacc-$version
	python --version
	export ver=`case $version in "3.6.4") echo 3.6 ;; "3.5.4") echo 3.5 ;; "3.5.0") echo 3.5 ;; "3.4.7") echo 3.4 ;; "3.3.7") echo 3.3 ;; "2.7.14") echo 2.7 ;; *) echo "invalid";; esac`
	export verstr=`case $ver in "3.6") echo "cp36-cp36m" ;; "3.5") echo "cp35-cp35m" ;; "3.4") echo "cp34-cp34m" ;; "3.3") echo "cp33-cp33m" ;; "2.7") echo "cp27-cp27mu" ;; *) echo "invalid";; esac`
 	python -m pip install --upgrade pip
	python -m pip install wheel
	export libPath=$(python -c "import distutils.util; print(distutils.util.get_platform())")
	echo $libPath
	export prefix="build\/lib."
	export suffix="-$ver"
	export arch=$(echo $libPath | sed -e "s/^$prefix//" -e "s/$suffix$//" | sed -e 's/-/_/g' | sed -e 's/\./_/g')
	echo $arch
	python -m pip install $HOME/xacc-wheelhouse/xacc-0.1.0-$verstr-$arch.whl
	git clone --recursive -b mccaskey/cpr_build https://github.com/ornl-qci/tnqvm
	cd tnqvm
        python setup.py build -t tmp_build --executable="/usr/bin/env python"

	export xaccdir=$(cd ../xacc && pwd)
	install_name_tool -change $xaccdir/tmp_build/lib/libcpr.dylib @rpath/libcpr.dylib build/lib.$libPath-$ver/xacc/plugins/libtnqvm.dylib
	install_name_tool -change $xaccdir/tmp_build/lib/libcpr.dylib @rpath/libcpr.dylib build/lib.$libPath-$ver/xacc/plugins/libtnqvm-itensor.dylib

	install_name_tool -change libboost_system.dylib @rpath/libboost_system.dylib build/lib.$libPath-$ver/xacc/plugins/libtnqvm.dylib
	install_name_tool -change libboost_filesystem.dylib @rpath/libboost_filesystem.dylib build/lib.$libPath-$ver/xacc/plugins/libtnqvm.dylib
	install_name_tool -change libboost_program_options.dylib @rpath/libboost_program_options.dylib build/lib.$libPath-$ver/xacc/plugins/libtnqvm.dylib
	install_name_tool -change libboost_regex.dylib @rpath/libboost_regex.dylib build/lib.$libPath-$ver/xacc/plugins/libtnqvm.dylib
	install_name_tool -change libboost_chrono.dylib @rpath/libboost_chrono.dylib build/lib.$libPath-$ver/xacc/plugins/libtnqvm.dylib
	install_name_tool -change libboost_graph.dylib @rpath/libboost_graph.dylib build/lib.$libPath-$ver/xacc/plugins/libtnqvm.dylib

	install_name_tool -change libboost_system.dylib @rpath/libboost_system.dylib build/lib.$libPath-$ver/xacc/plugins/libtnqvm-itensor.dylib
	install_name_tool -change libboost_filesystem.dylib @rpath/libboost_filesystem.dylib build/lib.$libPath-$ver/xacc/plugins/libtnqvm-itensor.dylib
	install_name_tool -change libboost_program_options.dylib @rpath/libboost_program_options.dylib build/lib.$libPath-$ver/xacc/plugins/libtnqvm-itensor.dylib
	install_name_tool -change libboost_regex.dylib @rpath/libboost_regex.dylib build/lib.$libPath-$ver/xacc/plugins/libtnqvm-itensor.dylib
	install_name_tool -change libboost_chrono.dylib @rpath/libboost_chrono.dylib build/lib.$libPath-$ver/xacc/plugins/libtnqvm-itensor.dylib
	install_name_tool -change libboost_graph.dylib @rpath/libboost_graph.dylib build/lib.$libPath-$ver/xacc/plugins/libtnqvm-itensor.dylib

	python setup.py bdist_wheel --skip-build
	mv dist/*.whl $HOME/tnqvm-wheelhouse
	python -m pip uninstall -y xacc
	source deactivate
	#rm -rf xacc
done

