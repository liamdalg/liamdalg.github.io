---
layout: post
title: Cython Is Cool
modified:
categories: 
description: A look at performance boosts with Cython.
tags: [python, project, optimization]
image:
  feature:
  credit:
  creditlink:
comments:
share:
date: 2018-12-25T04:33:01+00:00
---

A while back I started making a small image processing application in Python, but I got a little discouraged when I realised how slow it was at kernel convolutions. Recently, I decided to return to it and have a go at cleaning up the code and seeing if Cython could help me out.

But what is Cython? See [cython.org](https://cython.org/) for the full explanation. Glossing over a lot of details, Cython is a superset of Python which allows Python to easily interact with C functions and declare C-typed variables. It also has some other handy features, most notably the ability to generate code analysis of what each line of Python code is turned into.

# The Culprit

{% highlight python %}
def _kernel_convolution_2d(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    rows, cols = kernel.shape
    x, y = rows // 2, cols // 2
    # the kernel must be flipped else it is known as a 'correlation'
    flipped_kernel = np.fliplr(np.flipud(kernel))

    conv = np.empty(arr.shape)
    padded_arr = np.pad(arr, ((x, y), (x, y)), mode='constant')
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            conv[i][j] = np.sum(np.multiply(padded_arr[i:(i + rows), j:(j + cols)], flipped_kernel))

    return conv
{% endhighlight %}

<figure class="center">
	<img src="/images/2d-conv.gif" alt="">
	<figcaption><a href="http://deeplearning.stanford.edu/wiki/index.php/Feature_extraction_using_convolution" title="Example 2D Convolution">Example 2D convolution</a>.</figcaption>
</figure>

This is the function which performs a kernel convolution on a 2-dimensional Numpy array. The gif above demonstrates what it is doing: it 'slides' the kernel over the image centred around each pixel, multiplies each pixel element-wise with the kernel, and then takes the sum as the new pixel value. It does this by taking a slice of the array which is the same size as the kernel, multiplying them together with `np.multiply`, and then taking the sum with `np.sum`. Another quirk is that it pads the array with 0s to prevent the convolved array from being smaller than the source array, as displayed in the gif above.

Let's see if we can speed this up using Cython! Credit to [this](https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html#numpy-tutorial) tutorial.

The first thing to do is to create a copy of our code and make it into a `.pyx` file. We'll be using `cython` to compile into `.c` files, and then `gcc` to compile that into an `.so`.

{% highlight bash %}
cython <file>.pyx -3 -a 
gcc -shared -pthread -fPIC $(python3.5-config --cflags) -fno-strict-aliasing -o <outfile>.so <infile>.c
{% endhighlight %}

The `-3` flag makes sure to compile Python 3 rather than Python 2, and `-a` generates an `.html` file which we can use to check how the Python and C interact with each other. `gcc` requires `$(python3.5-config --cflags)` (replace 3.5 with your own version of Python) so that it knows where to find the bindings and flags relevant for Python.

<!--more-->

## Basic Types

Python is dynamically typed while C is statically typed, which can make for some pretty inefficient code since C cannot directly infer the types. Fortunately, we can use the `cdef` statement to enforce C types onto Python objects. The standard type for sizes is `Py_ssize_t`, and later we'll be using `long` which is analogous to `np.int64` for the array element values. While we're at it we can also explicitly unpack the `shape` tuples with the array `[]` operator since it is faster than tuple unpacking.

{% highlight python %}
cdef Py_ssize_t rows = kernel.shape[0]
cdef Py_ssize_t cols = kernel.shape[1]
cdef Py_ssize_t x = rows // 2
cdef Py_ssize_t y = cols // 2
{% endhighlight %}

Believe it or not, changing these lines has actually reduced the amount of code a lot. Without the explicit definition of types, C will attempt to type cast - this requires lots of error checking and boilerplate. Compiling with the `-a` flag generates a `.html` file which allows us to compare the C code generated:

{% highlight C %}
# Before
__pyx_t_1 = __Pyx_PyObject_GetAttrStr(__pyx_v_kernel, __pyx_n_s_shape); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 25, __pyx_L1_error)
__Pyx_GOTREF(__pyx_t_1);
__pyx_t_2 = __Pyx_GetItemInt(__pyx_t_1, 0, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 25, __pyx_L1_error)
__Pyx_GOTREF(__pyx_t_2);
__Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
__pyx_t_1 = __Pyx_PyObject_GetAttrStr(__pyx_v_kernel, __pyx_n_s_shape); if (unlikely(!__pyx_t_1)) __PYX_ERR(0, 25, __pyx_L1_error)
__Pyx_GOTREF(__pyx_t_1);
__pyx_t_3 = __Pyx_GetItemInt(__pyx_t_1, 1, long, 1, __Pyx_PyInt_From_long, 0, 0, 1); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 25, __pyx_L1_error)
__Pyx_GOTREF(__pyx_t_3);
__Pyx_DECREF(__pyx_t_1); __pyx_t_1 = 0;
__pyx_v_rows = __pyx_t_2;
__pyx_t_2 = 0;
__pyx_v_cols = __pyx_t_3;
__pyx_t_3 = 0;
__pyx_t_3 = __Pyx_PyInt_FloorDivideObjC(__pyx_v_rows, __pyx_int_2, 2, 0); if (unlikely(!__pyx_t_3)) __PYX_ERR(0, 26, __pyx_L1_error)
__Pyx_GOTREF(__pyx_t_3);
__pyx_t_2 = __Pyx_PyInt_FloorDivideObjC(__pyx_v_cols, __pyx_int_2, 2, 0); if (unlikely(!__pyx_t_2)) __PYX_ERR(0, 26, __pyx_L1_error)
__Pyx_GOTREF(__pyx_t_2);
__pyx_v_x = __pyx_t_3;
__pyx_t_3 = 0;
__pyx_v_y = __pyx_t_2;
__pyx_t_2 = 0;

# After
__pyx_v_rows = (__pyx_v_kernel.shape[0]);
__pyx_v_cols = (__pyx_v_kernel.shape[1]);
__pyx_v_x = __Pyx_div_Py_ssize_t(__pyx_v_rows, 2);
__pyx_v_y = __Pyx_div_Py_ssize_t(__pyx_v_cols, 2);
{% endhighlight %}

We can go even further by enforcing types on the `for i in range(...)` loops - if we don't, Cython doesn't know how to turn it directly into a C for-loop. Unfortunately `cdef` can only be used on the outermost block, so we need to define them before the loop runs. Once again, we can use the `Py_ssize_t` type for iterator variables and the maximum `range` value.

{% highlight python %}
cdef Py_ssize_t i = 0
cdef Py_ssize_t j = 0
cdef Py_ssize_t i_max = arr.shape[0]
cdef Py_ssize_t j_max = arr.shape[1]
for i in range(i_max):
    for j in range(j_max):

%timeit conv_old._kernel_convolution_2d(x, k)
156 ms ± 2.31 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
%timeit conv_cy._kernel_convolution_2d(x, k)
86.9 ms ± 1.25 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
{% endhighlight %}

## MemoryView

Looking at the benchmarks so far, we've reduced the time by a factor of 2, although lots of this has come from merely using C itself. How can we go lower? Enforcing types on the Numpy arrays is the next step, which we can do using a `MemoryView`. `MemoryView`s allow for access to a portion of memory with the `[]` operator much like a regular C array, without the expensiveness of doing it through Python. They also keep the safety that Python provides by storing some metadata about the memory alongside it. 

To define a 2-dimensional `MemoryView`, use `cdef long [:, :] array`. While we're at it we can enforce types on the formal parameters.

{% highlight python %}
def _kernel_convolution_2d(long[:, :] arr, long[:, :] kernel) -> np.ndarray:

    cdef Py_ssize_t rows = kernel.shape[0], cols = kernel.shape[1]
    cdef Py_ssize_t x = rows // 2, y = cols // 2

    cdef long[:, :] flipped_kernel = np.fliplr(np.flipud(kernel))

    cdef Py_ssize_t i = 0, j = 0
    cdef Py_ssize_t i_max = arr.shape[0]
    cdef Py_ssize_t j_max = arr.shape[1]

    cdef long[:, :] conv = np.empty((i_max, j_max), dtype=np.int64)
    cdef long[:, :] padded_arr = np.pad(arr, ((x, y), (x, y)), mode='constant')

    for i in range(i_max):
        for j in range(j_max):
            conv[i][j] = np.sum(np.multiply(padded_arr[i:(i + rows), j:(j + cols)], flipped_kernel))

    return np.asarray(conv)

%timeit -n10 conv_old._kernel_convolution_2d(x, k)
151 ms ± 2.96 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
%timeit -n10 conv_cy._kernel_convolution_2d(x, k)
233 ms ± 1.39 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
{% endhighlight %}

What's going on here? Shouldn't enforcing types on the array  make it faster? My best guess it has something to do with using `np.sum` and `np.multiply` - these functions work exclusively on Numpy arrays, which means we'll be converting `MemoryView` to an `ndarray` and vice-versa in each iteration of the for loop. In fact, looking at the C code that it generates, there are many calls to `__pyx_memoryview_fromslice` and many calls to `PyFunction`s. In other words, we're wasting time converting between Python and C types.

Instead, we can implement `np.sum` and `np.multiply` ourselves. 

{% highlight python %}
def _kernel_convolution_2d(long[:, :] arr, long[:, :] kernel) -> np.ndarray:

    cdef Py_ssize_t rows = kernel.shape[0], cols = kernel.shape[1]
    cdef Py_ssize_t x = rows // 2, y = cols // 2

    cdef long[:, :] flipped_kernel = np.fliplr(np.flipud(kernel))

    cdef Py_ssize_t i = 0, j = 0, k = 0, l = 0
    cdef Py_ssize_t i_max = arr.shape[0]
    cdef Py_ssize_t j_max = arr.shape[1]

    cdef long[:, :] conv = np.empty((i_max, j_max), dtype=np.int64)
    cdef long[:, :] padded_arr = np.pad(arr, ((x, y), (x, y)), mode='constant')

    cdef tmp = 0
    for i in range(i_max):
        for j in range(j_max):
            for k in range(rows):
                for l in range(cols):
                    tmp += padded_arr[i + k][j + l] * flipped_kernel[k][l]
            conv[i][j] = tmp
            tmp = 0

    return np.asarray(conv)

%timeit -n10 conv_old._kernel_convolution_2d(x, k)
195 ms ± 1.49 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
%timeit -n10 conv_cy._kernel_convolution_2d(x, k)
3.52 ms ± 1.05 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
{% endhighlight %}

That's a pretty substantial increase in performance! Approximately 56x in fact for a 128x128 array and 3x3 kernel. All we've done is enforce types on our variables and implemented the convolution logic explicitly. Obviously, a downside of this is that we lose the dynamic typing that Python provides, so this won't work for types which aren't `long` or `np.int64`. Fortunately I wanted to use this for images so `long` is more than enough.

## Extras

do later :)