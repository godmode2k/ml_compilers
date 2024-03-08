#!python ./postprocess.py

# Source-based:
# - https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html



import os.path
import numpy as np

from scipy.special import softmax

from tvm.contrib.download import download_testdata

# Download a list of labels
labels_url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
labels_path = download_testdata(labels_url, "synset.txt", module="data")

with open(labels_path, "r") as f:
    labels = [l.rstrip() for l in f]

output_file = "predictions.npz"

# Open the output and read the output tensor
if os.path.exists(output_file):
    with np.load(output_file) as data:
        print( data.files )

        #scores = softmax(data["output_0"])
        scores = softmax(data["data"])

        scores = np.squeeze(scores)
        ranks = np.argsort(scores)[::-1]

        for rank in ranks[0:5]:
            print("class='%s' with probability=%f" % (labels[rank], scores[rank]))
