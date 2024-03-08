/* --------------------------------------------------------------
Project:    TVM in C++ example
Purpose:
Author:     Ho-Jung Kim (godmode2k@hotmail.com)
Date:       Mar 7, 2024

License:

*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
-----------------------------------------------------------------
Note:
-----------------------------------------------------------------
Source-based:
 - https://github.com/apache/tvm/tree/main/apps/howto_deploy
 - https://github.com/apache/tvm/tree/main/apps/cpp_rtvm

Reference:
 - https://tvm.apache.org/docs/tutorial/introduction.html

Prerequisites:
    $ sudo mkdir /work
    $ sudo docker pull tlcpack/ci-cpu:20240105-165030-51bdaec6
    $ sudo docker run -it -v /work:/work tlcpack/ci-cpu:20240105-165030-51bdaec6

    // TVM
    // Get Source Code
    # cd /work
    # git clone --recursive https://github.com/apache/tvm.git
    # cd tvm

    // Build Configuration
    # mkdir build && cd build
    # cp ../cmake/config.cmake .
    (EDIT config.cmake)

    // build for release
    # cmake ..
    or
    // build for debug
    # cmake –DCMAKE_BUILD_TYPE=Debug ..
    # make -j4

    // TVM Environment
    # vim $HOME/.bashrc
    (EDIT $HOME/.bashrc) {
        export TVM_HOME=/work/tvm
        export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
    } EDIT $HOME/.bashrc
    # source $HOME/.bashrc


    // Install OpenCV
    # sudo apt-get install libopencv-dev


    # cd cpp-tvm/res

    // ONNX model
    # wget wget https://github.com/onnx/models/raw/b9a54e89508f101a1611cd64f4ef56b9cb62c7cf/vision/classification/resnet/model/resnet50-v2-7.onnx

    // compile model (outout: resnet50-tvm.so)
    # python compile_model.py

    // input image
    # wget https://s3.amazonaws.com/model-server/inputs/kitten.jpg

    // labels
    # wget https://s3.amazonaws.com/onnx-model-zoo/synset.txt

Build:
    // output: lib/...
    # sh run_example.sh

Run:
    // output: predictions.npz, results
    # LD_LIBRARY_PATH=/work/tvm/build:./3rdparty/cnpy/build ./lib/cpp_deploy_normal
    class='n02123045 tabby, tabby cat' with probability=0.513196
    class='n02123159 tiger cat' with probability=0.453955
    class='n02124075 Egyptian cat' with probability=0.027948
    class='n02129604 tiger, Panthera tigris' with probability=0.001375
    class='n02127052 lynx, catamount' with probability=0.001186

    // Comparing the results of CPP-TVM and postprocess.py

    // results from predictions.npz
    # python ./res/postprocess.py
    class='n02123045 tabby, tabby cat' with probability=0.513196
    class='n02123159 tiger cat' with probability=0.453955
    class='n02124075 Egyptian cat' with probability=0.027948
    class='n02129604 tiger, Panthera tigris' with probability=0.001375
    class='n02127052 lynx, catamount' with probability=0.001186
-------------------------------------------------------------- */



//! Header
// ---------------------------------------------------------------

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cstdio>
#include <string>
#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>
#include <cnpy.h>
#include <tvm/runtime/c_runtime_api.h>

#include <algorithm>
#include <cmath>



//! Definition
// ---------------------------------------------------------------

// Source-based: https://cloud.tencent.com/developer/article/2346575 (CV::Mat(HWC) to CHW)
#define SIZE_W 224
#define SIZE_D 224
void Mat_to_CHW(float* data, cv::Mat& frame) {
    //! preprocess.py: Normalize according to ImageNet
    float imagenet_mean[3] = { 0.485, 0.456, 0.406 };
    float imagenet_stddev[3] = { 0.229, 0.224, 0.225 };

    unsigned int volChl = SIZE_W * SIZE_D;
    for (int c = 0; c < 3; ++c) {
        for (unsigned j = 0; j < volChl; ++j) {
            //data[c * volChl + j] = static_cast<float>(float(frame.data[j * 3 + c]) / 255.);
            
            //! preprocess.py: Normalize according to ImageNet
            data[c * volChl + j] = static_cast<float>((float(frame.data[j * 3 + c]) / 255. - imagenet_mean[c]) / imagenet_stddev[c]);
        }
    }
}

inline size_t GetMemSize(tvm::runtime::NDArray& narr) {
  size_t size = 1;
  for (tvm_index_t i = 0; i < narr->ndim; ++i) {
    size *= static_cast<size_t>(narr->shape[i]);
  }
  size *= (narr->dtype.bits * narr->dtype.lanes + 7) / 8;
  return size;
}

// Source: https://mawile.tistory.com/205
// https://www.HostMath.com/Show.aspx?Code=f(sj)%20%3D%20%5Cfrac%7Be%5E%7Bsj%7D%7D%7B%5Csum_%7Bi%3D1%7D%5E%7Bm%7De%5E%7Bsi%7D%7D
template<typename dataType>
double softmax_s(std::vector<dataType>& arr, dataType sj){
	if(!std::any_of(arr.begin(), arr.end(), [&sj](dataType& j){ return j == sj; })) throw std::runtime_error("Invalid value");
	
	dataType maxElement = *std::max_element(arr.begin(), arr.end());
	double sum = 0.0;
	for(auto const& i : arr) sum += std::exp(i - maxElement);
	
	return (std::exp(sj - maxElement) / sum);
}
/*
// softmax example
int main() {
	std::vector<double> arr_s = { 1980, 1990, 2000 };

	auto sv_1980 = softmax_s<double>(arr_s, 1980);
	auto sv_1990 = softmax_s<double>(arr_s, 1990);
	auto sv_2000 = softmax_s<double>(arr_s, 2000);
	auto sv_all = softmax_s<double>(arr_s, 1980) + softmax_s<double>(arr_s, 1990) + softmax_s<double>(arr_s, 2000);

	std::cout << "sv_1980: " << sv_1980 << '(' << double(sv_1980 * 100) << "%)\n";
	std::cout << "sv_1990: " << sv_1990 << '(' << double(sv_1990 * 100) << "%)\n";
	std::cout << "sv_2000: " << sv_2000 << '(' << double(sv_2000 * 100) << "%)\n";
	std::cout << "sv_all: " << sv_all << '(' << double(sv_all * 100) << "%)\n\n";
}
*/



//! Prototype
// ---------------------------------------------------------------



//! Implementation
// ---------------------------------------------------------------

void test(void) {
    LOG(INFO) << "Running graph executor...";

    // load in the library
    DLDevice dev{ kDLCPU, 0 };

    // module compiled for TVM (1000 labels)
    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("./res/resnet50-tvm.so");
    const int output_size = 1000;
    // SEE: labels file "res/synset.txt"

    // create the graph executor module
    tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
    tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
    tvm::runtime::PackedFunc run = gmod.GetFunction("run");


    // NDArray for input, output
    uint8_t dtype_code = kDLFloat;
    uint8_t dtype_bits = 32;
    uint8_t dtype_lanes = 1;
    // SEE: 'shape_dict = { input_name: (1, 3, 224, 224) }' in res/compile_model.py
    tvm::runtime::NDArray x = tvm::runtime::NDArray::Empty({1, 3, 224, 224}, DLDataType{dtype_code, dtype_bits, dtype_lanes}, dev);
    tvm::runtime::NDArray y = tvm::runtime::NDArray::Empty({1, output_size}, DLDataType{dtype_code, dtype_bits, dtype_lanes}, dev);



    // preprocess
    // -----------------------------------------------------------
    // Load an input image
    cv::Mat img = cv::imread("./res/kitten.jpg");
    cv::Mat frame;
    cv::Mat input;
    cv::cvtColor(img, frame, cv::COLOR_BGR2RGB);
    //cv::cvtColor(img, frame, cv::COLOR_BGR2GRAY);
    cv::resize(frame, input, cv::Size(SIZE_W, SIZE_D));

    float data[SIZE_W * SIZE_D * 3];
    Mat_to_CHW(data, input);
    memcpy(x->data, &data, SIZE_W * SIZE_D * 3 * sizeof(float));

    /*
    // Load an input image from .npz file
    {
        // Source: https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html
        // $ python ./preprocess.py
        // output: imagenet_cat.npz

        //load a single var from the npz file
        //cnpy::NpyArray arr = cnpy::npz_load("imagenet_cat.npz", "data");
        //float* _data = arr.data<float>();

        //load the entire npz file
        cnpy::npz_t my_npz = cnpy::npz_load("imagenet_cat.npz");
        cnpy::NpyArray arr = my_npz["data"];
        float* _data = arr.data<float>();

        memset(x->data, 0x00, SIZE_W * SIZE_D * 3 * sizeof(float));
        memcpy(x->data, _data, SIZE_W * SIZE_D * 3 * sizeof(float));
    }
    */
    // -----------------------------------------------------------



    // set the right input
    //set_input("x", x);
    // SEE: 'input_name' in res/compile_model.py
    set_input("data", x);

    // run the code
    run();

    // get the output
    get_output(0, y);



    // print results: limit 3
    /*
    {
        auto result = static_cast<float*>(y->data);
        for (int i = 0; i < 3; i++) { LOG(INFO) << result[i]; }
    }
    */



    // Save the output as .npz file
    // -----------------------------------------------------------
    std::string outputfile = "predictions.npz";
    {
        tvm::runtime::NDArray out_arr = y;

        auto ssize = GetMemSize(out_arr);
        LOG(INFO) << "Output Size: " << ssize << "  bytes";

        void* data = (void*)malloc(ssize * (out_arr->dtype.bits * out_arr->dtype.lanes + 7) / 8);
        out_arr.CopyToBytes(data, ssize);
        std::vector<size_t> shape;

        for (int j = 0; j < out_arr->ndim; ++j) shape.push_back(out_arr->shape[j]);

        LOG(INFO) << "save to npz: " << outputfile;
        cnpy::npz_save<float>(outputfile, "data", (float*)data, shape, "w"); // (int8_t*)data
        free(data);
    }
    // -----------------------------------------------------------



    // postprocess
    // -----------------------------------------------------------
    // postprocess.py
    //labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
    //labels = ...
    //scores = softmax(data["data"])
    //scores = np.squeeze(scores)
    //ranks = np.argsort(scores)[::-1]
    //for rank in ranks[0:5]:
    //    print("class='%s' with probability=%f" % (labels[rank], scores[rank]))

    auto scores = static_cast<float*>(y->data);
    std::vector<double> scores_vec;
    typedef struct { double softmax_val; int index; } ranks_st;
    std::list<ranks_st> ranks_list;
    std::vector<char*> labels_vec;

    LOG(INFO) << "scores length: " << output_size;
    

    // softmax
    for ( int i = 0; i < output_size; i++ ) {
        scores_vec.push_back( static_cast<double>(scores[i]) );
    }
    for ( int i = 0; i < output_size; i++ ) {
        double val = softmax_s<double>( scores_vec, scores_vec[i] );
        //val = double( val * 100 );
        ranks_list.push_back( ranks_st {val, i} );
        //LOG(INFO) << i << ", score = " << scores[i]  << ", rank = " << val;
    }


    // numpy.argsort
    // asc: ranks.sort(), desc: ranks.sort( std::greater<double>() )
    ranks_list.sort( [](ranks_st& a, ranks_st& b) { return a.softmax_val > b.softmax_val; } );


    // load a labels file
    const char* labels_filename = "res/synset.txt";
    std::FILE* fp_labels = NULL;
    if ( (fp_labels = fopen(labels_filename, "r")) == NULL ) {
        LOG(INFO) << "Cannot open labels file: " << labels_filename;
        return;
    }
    
    for ( char buf[255]; std::fgets(buf, sizeof(buf), fp_labels) != nullptr; ) {
        buf[strlen(buf)-1] = '\0';
        //LOG(INFO) << buf;
        char* val = new char[255];
        memcpy( val, (char*)&buf, 255 );
        labels_vec.push_back( val );
    }
    std::fclose( fp_labels );


    if ( ranks_list.empty() ) {
        LOG(INFO) << "ranks size (empty): " << ranks_list.size();
        return;
    }


    // results
    std::list<ranks_st>::iterator it = ranks_list.begin();
    for ( int i = 0; i < 5; i++ ) {
        int index = (*it).index;
        double softmax_val = (*it).softmax_val;
        LOG(INFO) << "index = " << index << ", class = '" << labels_vec[index]
            << "' with probability = " << softmax_val;
        ++it;
    }


    // release
    std::vector<char*>::iterator it_labels;
    for ( it_labels = labels_vec.begin(); it_labels != labels_vec.end(); ++it_labels ) {
        char* val = (*it_labels); delete val;
    }
    labels_vec.clear();
    // -----------------------------------------------------------
}



//! MAIN
// ---------------------------------------------------------------
int main(void) {
  test();


  return 0;
}



