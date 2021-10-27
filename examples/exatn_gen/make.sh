mpicxx -std=c++17 -I$HOME/.xacc/include/xacc -I$HOME/.xacc/include/quantum/gate -I$HOME/.xacc/include/cppmicroservices4 -c main_asp.cpp -o main_asp.o
mpicxx main_asp.o -Wl,-rpath,$HOME/.xacc/lib -L$HOME/.xacc/lib -lxacc -lxacc-quantum-gate -o main_asp.x

mpicxx -std=c++17 -I$HOME/.xacc/include/xacc -I$HOME/.xacc/include/quantum/gate -I$HOME/.xacc/include/cppmicroservices4 -c main_qaoa.cpp -o main_qaoa.o
mpicxx main_qaoa.o -Wl,-rpath,$HOME/.xacc/lib -L$HOME/.xacc/lib -lxacc -lxacc-quantum-gate -o main_qaoa.x
