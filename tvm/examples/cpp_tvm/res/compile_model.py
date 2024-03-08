# Source-based:
# - https://github.com/apache/tvm/blob/main/apps/howto_deploy/prepare_test_libs.py
#
# Reference:
# - https://tvm.apache.org/docs/tutorial/tvmc_python.html
# - https://tvm.apache.org/docs/tutorial/autotvm_relay_x86.html
# - https://github.com/apache/tvm/issues/16607



import tvm
import numpy as np
from tvm import te
from tvm import relay
import os

import onnx
from tvm.driver import tvmc
import sys



def compile_model():

    #_onnx_model = tvmc.load( '../../../resnet50-v2-7.onnx' )
    #_onnx_model.summary()
    #sys.exit()

    # ONNX model file
    # https://github.com/onnx/models/raw/b9a54e89508f101a1611cd64f4ef56b9cb62c7cf/vision/classification/resnet/model/resnet50-v2-7.onnx
    # labels
    # https://s3.amazonaws.com/onnx-model-zoo/synset.txt

    onnx_model = onnx.load( './resnet50-v2-7.onnx' )
    #input_name = 'input.1'
    input_name = 'data'
    shape_dict = { input_name: (1, 3, 224, 224) }

    mod, params = relay.frontend.from_onnx( onnx_model, shape_dict )
    target = tvm.target.Target( "llvm", host='llvm' )

    with tvm.transform.PassContext( opt_level=3 ):
        lib = relay.build( mod, target=target, params=params )
    dev = tvm.cpu()

    path_lib = os.path.join( os.getcwd(), "resnet50-tvm.so" )
    lib.export_library( path_lib )



if __name__ == "__main__":
    compile_model()



