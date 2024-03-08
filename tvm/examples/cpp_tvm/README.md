# TVM in C++ examples


Summary
----------
> TVM in C++ </br>
> - Compile models (onnx model: resnet50-v2-7.onnx) and Run </br>
> </br>
> WORK IN-PROGRESS


Environment
----------
> build all and tested on GNU/Linux

    GNU/Linux: Ubuntu 20.04_x64 LTS
    Docker
    TVM build environment (docker image): tlcpack/ci-cpu (dev tools)
    TVM v0.16.dev0
    OpenCV 4.x


## Reference

```sh
https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html
https://tvm.apache.org/docs/tutorial/tvmc_python.html
#
https://github.com/apache/tvm/tree/main/apps/howto_deploy
https://github.com/apache/tvm/tree/main/apps/cpp_rtvm

tvm 문서 한국어 번역: https://github.com/godmode2k/ml_compilers/tree/main/tvm/tvm_docs_ko.pdf
```


## Build and Run

```sh
// images
https://hub.docker.com/r/tlcpack/ci-cpu/tags


// Pull Docker Image and Run
$ sudo mkdir /work
$ sudo docker pull tlcpack/ci-cpu:20240105-165030-51bdaec6
$ sudo docker run -it -v /work:/work tlcpack/ci-cpu:20240105-165030-51bdaec6


// Get Source Code
(container)# cd /work
(container)# apt-get update && apt-get install libopencv-dev
(container)# git clone --recursive https://github.com/apache/tvm.git
(container)# cd tvm


// Build Configuration
(container)# mkdir build && cd build
(container)# cp ../cmake/config.cmake .
(container)# vim config.cmake
(EDIT config.cmake) {
    set(USE_CUDA OFF) or set(USE_CUDA ON)

    set(USE_GRAPH_EXECUTOR ON)
    set(USE_PROFILER ON)

    # debug with IRs, set(USE_RELAY_DEBUG ON)
    set(USE_RELAY_DEBUG OFF) or set(USE_RELAY_DEBUG ON)

    # set environment variable TVM_LOG_DEBUG.
    # export TVM_LOG_DEBUG="ir/transform.cc=1,relay/ir/transform.cc=1"

    # enable llvm with cmake's find search
    set(USE_LLVM OFF) -> set(USE_LLVM ON)
    # or set path
    #set(USE_LLVM OFF) -> set(USE_LLVM /path/to/llvm-config)
} EDIT config.cmake


// build for release
(container)# cmake ..

or

// build for debug
(container)# cmake –DCMAKE_BUILD_TYPE=Debug ..
(container)# make -j4


// TVM Environment
(container)# vim $HOME/.bashrc
(EDIT $HOME/.bashrc) {
    export TVM_HOME=/work/tvm
    export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
} EDIT $HOME/.bashrc
(container)# source $HOME/.bashrc

or run a shell script below.

(container)# cat add_tvm_env_path.sh
#!/bin/sh
unset TVM_HOME
unset PYTHONPATH
echo 'export TVM_HOME=/work/tvm' >> $HOME/.bashrc
echo 'export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}' >> $HOME/.bashrc
source $HOME/.bashrc

(container)# source add_tvm_env_path.sh


// Run TVMC
(container)# python -m tvm.driver.tvmc


----------


// C++ example

// Get Source Code
(container)# cd /work
(container)# git clone https://github.com/godmode2k/ml_compilers.git


// Build Configuration
(container)# cd ml_compilers/tvm/examples/cpp-tvm
(container)# vim Makefile
(EDIT Makefile) {
    TVM_ROOT=$(shell cd /work/tvm; pwd)
    or
    TVM_ROOT=/work/tvm
} EDIT Makefile


// OpenCV: skip if you installed already
(container)# apt-get update && apt-get install libopencv-dev

(container)# cd res

// ONNX model
(container)# wget https://github.com/onnx/models/raw/b9a54e89508f101a1611cd64f4ef56b9cb62c7cf/vision/classification/resnet/model/resnet50-v2-7.onnx

// compile model (outout: resnet50-tvm.so)
(container)# python compile_model.py

// input image
(container)# wget https://s3.amazonaws.com/model-server/inputs/kitten.jpg
// labels
(container)# wget https://s3.amazonaws.com/onnx-model-zoo/synset.txt


// build all (output: lib/...)
(container)# cd ..
(container)# sh run_example.sh


// run
// output: predictions.npz, results
(container)# LD_LIBRARY_PATH=/work/tvm/build:./3rdparty/cnpy/build ./lib/cpp_deploy_normal
Running graph executor...
Output Size: 4000  bytes
save to npz: predictions.npz
scores length: 1000
index = 281, class = 'n02123045 tabby, tabby cat' with probability = 0.513196
index = 282, class = 'n02123159 tiger cat' with probability = 0.453955
index = 285, class = 'n02124075 Egyptian cat' with probability = 0.0279478
index = 292, class = 'n02129604 tiger, Panthera tigris' with probability = 0.00137514
index = 287, class = 'n02127052 lynx, catamount' with probability = 0.00118606

// Comparing the results of CPP-TVM and postprocess.py

// postprocess
// results from predictions.npz
(container)# python ./res/postprocess.py
class='n02123045 tabby, tabby cat' with probability=0.513196
class='n02123159 tiger cat' with probability=0.453955
class='n02124075 Egyptian cat' with probability=0.027948
class='n02129604 tiger, Panthera tigris' with probability=0.001375
class='n02127052 lynx, catamount' with probability=0.001186
```



