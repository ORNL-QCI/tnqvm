[![Build Status](https://travis-ci.org/maddouri/hyper_array.svg?branch=master)](https://travis-ci.org/maddouri/hyper_array)

# hyper_array

`hyper_array::array` is a simple, templated, multi-dimensional array class that is inspired by [`orca_array`](https://github.com/astrobiology/orca_array). It uses modern C++ techniques in order to achieve good performance, code clarity and platform independence.

`hyper_array` is header-only, contained in a single file -- [`hyper_array.hpp`](include/hyper_array/hyper_array.hpp) -- and doesn't depend on external libraries. Its only requirement is a C++11-compliant compiler.

<!--
grep -e '^#\+' README.md | sed 's/#\{3\}/    */' | sed 's/#\{2\}/  */' | sed 's/#\{1\}/*/' | sed 's/^\([^A-Za-z]\+\)\(.\+\)$/\1[\2](#\2)/'
-->

  * [Basics](#basics)
    * [Main Class](#main-class)
    * [Array Order](#array-order)
    * [Construction](#construction)
    * [Assignment](#assignment)
    * [Element Access](#element-access)
    * [Standard Library Compatibility](#standard-library-compatibility)
  * [Development](#development)



## Basics

### Main Class

The class template `hyper_array::array<ValueType, Dimensions, Order>` represents a `Dimensions`-dimension array of `ValueType` elements. Therefore, the type and number of dimensions are specified at compile-time. The length along each dimension can be set at run-time.

### Array Order

The third template argument --`Order`-- designates the [array order](https://en.wikipedia.org/wiki/Row-major_order) which can be either **row-major** _(i.e. C convention)_ or **column-major** _(i.e. Fortran convention)_ (cf. `hyper_array::array_order` enum).

By default, if `Order` is not specified, the order is row-major.

### Construction

A new array can be instantiated using one of the following constructors:

```c++
#include "hyper_array/hyper_array.hpp"
using hyper_array::array;

/// the usual way of constructing hyper arrays
array(DimensionLengths... dimensionLengths);
// usage example
array<double, 3> my3DArray{32, 64, 128};

/// copy constructor
array(const array_type& other);
// usage example
auto arrayCopy = my3DArray;

/// move constructor
array(array_type&& other);
// usage example
auto consumer = std::move(my3DArray);

/// create a new hyper array from "raw data"
array(::std::array<size_type, Dimensions> lengths, value_type* rawData);
// usage example
double* rawData = new double[262144];
array<double, 3> dataWrapper{{32, 64, 128}, rawData};

/// create and initialize a hyper array
/// given the dimension lengths and the array elements
array(::std::array<size_type, Dimensions> lengths,  // dimension lenths
      std::initializer_list<value_type>   values,   // array elements (you can provide less than size() elements)
      const value_type& defaultValue      = {});    // default initialization value (in case values.size() < size())
// usage example
const array<double, 2> constantArray{
    {2, 3},       // dimension lengths
    {11, 12, 13,
     21, 22, 23}  // array elements
};

```

### Assignment

`hyper_array` can be assigned for performing both deep copying and moving:

```c++
array<double, 4> hyperArray{42, 42, 42, 42};

/// copy assignment
array_type& operator=(const array_type& other);
// usage example
array<double, 4> someArray{32, 64, 128, 256};
// ...
hyperArray = someArray;

/// move assignment
array_type& operator=(array_type&& other);
// usage example
array<double, 4> temporaryArray{32, 64, 128, 256};
// ...
hyperArray = std::move(temporaryArray);
```

### Element Access

Single elements can be accessed for reading and assignment using various methods:

```c++
/// access using an index tuple
      value_type& operator()(Indices... indices);
const value_type& operator()(Indices... indices) const;
// usage example
array<double, 3> arr{4, 5, 6};
arr(3, 1, 4) = 3.14;
std::cout << "arr(3, 1, 4) == " << arr(3, 1, 4) << std::endl;  // arr(3, 1, 4) = 3.14

/// access using an index tuple, with index bounds checking
      value_type& at(Indices... indices);
const value_type& at(Indices... indices) const;
// usage example
array<double, 3> arr{4, 5, 6};
arr.at(3, 1, 4) = 3.14;
std::cout << "arr.at(3, 1, 4) == " << arr.at(3, 1, 4) << std::endl;  // arr.at(3, 1, 4) = 3.14

/// access using the index of the element in the underlying data array
      value_type& operator[](const index_type idx);
const value_type& operator[](const index_type idx) const;
// usage example
array<int, 3> arr{4, 5, 6};
arr[100] = 314;
cout << "arr[100] == arr(3, 1, 4): " << std::boolalpha << (arr[100] == arr(3, 1, 4)) << endl;  // arr[100] == arr(3, 1, 4): true
```

### Standard Library Compatibility

Currently, `hyper_array::array` implements the same iterators as `std::array`, which makes it compatible with most of the C++ Standard Library's algorithms and containers, as well as the range-based for loop syntax introduced in C++11. In addition, `operator<<()` is overloaded in order to provide an easy way to visualize array information.

```c++

// range-based for loop
array<double, 3> aa{4, 5, 6};
for (auto& x : aa) {
    x = 1;  // initialize the array
}

// algorithms
std::iota(aa.begin(), aa.end(), 1);            // fill aa with a sequence of consecutive numbers
array<double, 3> bb{aa.lengths()};             // array::lengths() returns the lengths along each direction
std::copy(aa.begin(), aa.end(), bb.rbegin());  // reverse-copy
array<double, 3> cc{aa.lengths()};
// cc = aa + bb
std::transform(aa.begin(), aa.end(),
               bb.begin(),
               cc.begin(),
               [](double a, double b) { return a + b; });

// stream operator
cout << "cc: " << cc << endl;
// cc: [dimensions: 3 ][lengths: 4 5 6 ][coeffs: 30 6 1 ][size: 120 ][data: 121 121 121 ...]
```

## Development

`hyper_array` is in constant development and new features will be added when appropriate. The goal is to keep it simple but useful, and efficient but maintainable.
