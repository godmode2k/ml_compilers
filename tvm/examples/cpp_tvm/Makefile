# Source-based:
# - https://github.com/apache/tvm/tree/main/apps/howto_deploy



# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.



# Makefile Example to deploy TVM modules.
#TVM_ROOT=$(shell cd ../..; pwd)
#DMLC_CORE=${TVM_ROOT}/3rdparty/dmlc-core

TVM_ROOT=$(shell cd ../../../tvm_v0.16.dev0; pwd)

PKG_CFLAGS = -std=c++17 -O2 -fPIC\
	-I${TVM_ROOT}/include\
	-I./3rdparty/dmlc-core/include\
	-I./3rdparty/dlpack/include\
	-DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\>

PKG_LDFLAGS = -L${TVM_ROOT}/build -ldl -pthread


# OpenCV4, from source build
#OPENCV_ROOT=$(shell cd ../libs/opencv-4.9.0; pwd)
#PKG_CFLAGS += \
-I${TVM_ROOT} \
-I${OPENCV_ROOT}/include \
-I${OPENCV_ROOT}/build \
-I${OPENCV_ROOT}/modules/calib3d/include \
-I${OPENCV_ROOT}/modules/core/include \
-I${OPENCV_ROOT}/modules/dnn/include \
-I${OPENCV_ROOT}/modules/features2d/include \
-I${OPENCV_ROOT}/modules/flann/include \
-I${OPENCV_ROOT}/modules/gapi/include \
-I${OPENCV_ROOT}/modules/highgui/include \
-I${OPENCV_ROOT}/modules/imgcodecs/include \
-I${OPENCV_ROOT}/modules/imgproc/include \
-I${OPENCV_ROOT}/modules/java/include \
-I${OPENCV_ROOT}/modules/js/include \
-I${OPENCV_ROOT}/modules/ml/include \
-I${OPENCV_ROOT}/modules/objc/include \
-I${OPENCV_ROOT}/modules/objdetect/include \
-I${OPENCV_ROOT}/modules/photo/include \
-I${OPENCV_ROOT}/modules/python/include \
-I${OPENCV_ROOT}/modules/stitching/include \
-I${OPENCV_ROOT}/modules/ts/include \
-I${OPENCV_ROOT}/modules/video/include \
-I${OPENCV_ROOT}/modules/videoio/include \
-I${OPENCV_ROOT}/modules/world/include
#
#PKG_LDFLAGS += \
-L${OPENCV_ROOT}/build_res/a \
-L../cpp_rtvm/3rdparty/cnpy/build -lcnpy -lz \
-lopencv_highgui -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_imgproc -lopencv_core -lopencv_imgcodecs
#-lopencv_dnn -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_video -lopencv_videoio
#-lopencv_gapi -lopencv_ts 


# OpenCV4
# $ sudo apt-get install libopencv-dev
PKG_CFLAGS += \
-I${TVM_ROOT} \
-I../cpp_rtvm/3rdparty/cnpy \
-I/usr/include/opencv4 \
-g
#
PKG_LDFLAGS += \
-L../cpp_rtvm/3rdparty/cnpy/build -lcnpy -lz \
-lopencv_highgui -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_imgproc -lopencv_core -lopencv_imgcodecs
#-lopencv_dnn -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_video -lopencv_videoio



.PHONY: clean all

all: lib/cpp_deploy_pack lib/cpp_deploy_normal

# Build rule for all in one TVM package library
lib/libtvm_runtime_pack.o: tvm_runtime_pack.cc
	@mkdir -p $(@D)
	$(CXX) -c $(PKG_CFLAGS) -o $@  $^


# The code library built by TVM
#lib/test_addone_sys.o: prepare_test_libs.py
#	@mkdir -p $(@D)
#	python3 prepare_test_libs.py
#
# Deploy using the all in one TVM package library
#lib/cpp_deploy_pack: cpp_deploy.cc lib/test_addone_sys.o lib/libtvm_runtime_pack.o
#	@mkdir -p $(@D)
#	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)
#
# Deploy using pre-built libtvm_runtime.so
#lib/cpp_deploy_normal: cpp_deploy.cc lib/test_addone_sys.o
#	@mkdir -p $(@D)
#	$(CXX) $(PKG_CFLAGS) -o $@  $^ -ltvm_runtime $(PKG_LDFLAGS)


# Deploy using the all in one TVM package library
lib/cpp_deploy_pack: cpp_deploy.cc lib/libtvm_runtime_pack.o
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ $(PKG_LDFLAGS)

# Deploy using pre-built libtvm_runtime.so
lib/cpp_deploy_normal: cpp_deploy.cc
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -o $@  $^ -ltvm_runtime $(PKG_LDFLAGS)


clean:
	rm -rf lib
