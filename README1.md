# TensorFlow Launcher

It is a simple tool to run network topology over the given set of images or binary files using TensorFlow framework.

## Install prerequisites

### TensorFlow 1.15 (deprecated)
1. Install bazel [0.25.2](https://github.com/bazelbuild/bazel/releases/download/0.25.2/bazel-0.25.2-installer-linux-x86_64.sh)
    ```
    chmod +x bazel-0.25.2-installer-linux-x86_64.sh
    ./bazel-0.25.2-installer-linux-x86_64.sh --prefix=<path_to_install>
    ```
2. Install tensorflow requirements
    ```
    sudo apt-get install autoconf                                                  \
                         libtool                                                   \
                         mlocate                                                   \
                         zlib1g-dev
    ```
    Install the dependencies which are listed in the [setup.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py) file under REQUIRED_PACKAGES

3. Clone repository
    ```
    git clone https://github.com/tensorflow/tensorflow.git --recursive -b v1.15.2
    ```
4. Patch code to use contribs and right protobuf

    Apply the following patch
    ```
    --- a/tensorflow/BUILD
    +++ b/tensorflow/BUILD
    @@ -594,6 +594,10 @@ tf_cc_shared_object(
             "//tensorflow/cc:scope",
             "//tensorflow/cc/profiler",
             "//tensorflow/core:tensorflow",
    +        "//tensorflow/contrib/rnn:all_ops",
    +        "//tensorflow/contrib/rnn:all_kernels",
    +        "//tensorflow/contrib/seq2seq:beam_search_ops_op_lib",
    +        "//tensorflow/contrib/seq2seq:beam_search_ops_kernels"
         ] + if_ngraph(["@ngraph_tf//:ngraph_tf"]),
     )

    ```
    for build with TensorRT:
    ```
    --- a/tensorflow/BUILD
    +++ b/tensorflow/BUILD
    @@ -668,6 +668,13 @@ tf_cc_shared_object(
             "//tensorflow/cc:scope",
             "//tensorflow/cc/profiler",
             "//tensorflow/core:tensorflow",
    +        "//tensorflow/contrib/rnn:all_ops",
    +        "//tensorflow/contrib/rnn:all_kernels",
    +        "//tensorflow/contrib/seq2seq:beam_search_ops_op_lib",
    +        "//tensorflow/contrib/seq2seq:beam_search_ops_kernels",
    +        "//tensorflow/contrib/tensorrt:trt_conversion",
    +        "//tensorflow/contrib/tensorrt:trt_op_kernels",
    +        "//tensorflow/contrib/tensorrt:trt_engine_op_op_lib"
         ] + if_ngraph(["@ngraph_tf//:ngraph_tf"]),
     )

    ```
5. Download dependecies
    ```
    cd tensorflow
    ./tensorflow/contrib/makefile/download_dependencies.sh

    ```
6. Set environment variables
    ```
    export CC_OPT_FLAGS="-march=native"
    export TF_NEED_GCP=0
    export TF_NEED_HDFS=0
    export TF_NEED_OPENCL=0
    export TF_NEED_OPENCL_SYCL=0
    export TF_NEED_CUDA=1
    export TF_NEED_TENSORRT=0
    export TF_NEED_JEMALLOC=1
    export TF_NEED_VERBS=0
    export TF_NEED_MPI=0
    export TF_ENABLE_XLA=1
    export TF_NEED_S3=0
    export TF_NEED_GDR=0
    export TF_CUDA_CLANG=0
    export TF_SET_ANDROID_WORKSPACE=0
    export TF_DOWNLOAD_CLANG=0
    export TF_NEED_KAFKA=0
    export TF_NEED_IGNITE=0
    export TF_NEED_ROCM=0
    export PYTHON_BIN_PATH="$(which python3)"
    export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'import site; print(site.getsitepackages()[0])')"
    export http_proxy=http://proxy-chain.intel.com:911
    export https_proxy=http://proxy-chain.intel.com:912
    export TF_CUDA_COMPUTE_CAPABILITIES="6.1"
    export GCC_HOST_COMPILER_PATH="/usr/bin/gcc"
    ```
    For build with TensorRT:
    ```
    export TF_NEED_TENSORRT=1
    ```
7. Build
    ```
    cd <tensorflow-root-dir>
    ./configure
    bazel build --config=mkl -c opt                               \
                --copt=${CC_OPT_FLAGS}                            \
               tensorflow:libtensorflow_cc.so                     \
               tensorflow:libtensorflow_framework.so              \
               tensorflow/install_headers                         \
               tensorflow/tools/graph_transforms:transform_utils

    bazel shutdown
    ```
8. Build and install python
    ```
    bazel  build --config=mkl -c opt \
                --copt=${CC_OPT_FLAGS} \
                //tensorflow/tools/pip_package:build_pip_package
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package  /tmp/tensorflow_pkg
    sudo pip3 install /tmp/tensorflow_pkg/tensorflow*.whl
    ```


### Or TensorFlow 2.5.0
1. Install bazel [3.7.2](https://github.com/bazelbuild/bazel/releases/download/3.7.2/bazel-3.7.2-installer-linux-x86_64.sh)
    ```
    chmod +x bazel-3.7.2-installer-linux-x86_64.sh
    ./bazel-3.7.2-installer-linux-x86_64.sh --prefix=<path_to_install>
    ```
2. Install python protobuf
    ```
    sudo pip3 install protobuf==3.12.4
    ```
3. Clone repository
    ```
    git clone https://github.com/tensorflow/tensorflow.git --recursive -b v2.5.0
    ```
4. Set environment variables
    ```
    export CC_OPT_FLAGS="-march=native"
    export TF_NEED_GCP=0
    export TF_NEED_HDFS=0
    export TF_NEED_OPENCL=0
    export TF_NEED_OPENCL_SYCL=0
    export TF_NEED_CUDA=1
    export TF_NEED_TENSORRT=0
    export TF_NEED_JEMALLOC=1
    export TF_NEED_VERBS=0
    export TF_NEED_MPI=0
    export TF_ENABLE_XLA=1
    export TF_NEED_S3=0
    export TF_NEED_GDR=0
    export TF_CUDA_CLANG=0
    export TF_SET_ANDROID_WORKSPACE=0
    export TF_DOWNLOAD_CLANG=0
    export TF_NEED_KAFKA=0
    export TF_NEED_IGNITE=0
    export TF_NEED_ROCM=0
    export PYTHON_BIN_PATH="$(which python3)"
    export PYTHON_LIB_PATH="$($PYTHON_BIN_PATH -c 'import site; print(site.getsitepackages()[0])')"
    export http_proxy=http://proxy-chain.intel.com:911
    export https_proxy=http://proxy-chain.intel.com:912
    export TF_CUDA_COMPUTE_CAPABILITIES="6.1"
    export GCC_HOST_COMPILER_PATH="/usr/bin/gcc"
    ```
    For build with TensorRT:
    ```
    export TF_NEED_TENSORRT=1
    ```
5. Build
    ```
    cd <tensorflow-root-dir>
    ./configure
    bazel build --config=mkl -c opt                               \
                --copt=${CC_OPT_FLAGS}                            \
               tensorflow:libtensorflow_cc.so                     \
               tensorflow:libtensorflow_framework.so              \
               tensorflow/install_headers                         \
               tensorflow/tools/graph_transforms:transform_utils

    bazel shutdown
    ```
6. Build and install python
    ```
    bazel  build --config=mkl -c opt \
                --copt=${CC_OPT_FLAGS} \
                //tensorflow/tools/pip_package:build_pip_package
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package  /tmp/tensorflow_pkg
    sudo pip3 install /tmp/tensorflow_pkg/tensorflow*.whl
    ```

### TensorFlow-ONNX
```
git clone https://github.com/onnx/tensorflow-onnx.git -b v1.9.3
cd tensorflow-onnx
python3 setup.py install
```

## Build tools including TensorFlow launcher
```
cd <model-zoo-tools-build>
cmake -DBUILD_TF_LAUNCHER=<ON, OFF>                                            \
      -DBUILD_TENSORRT_TF_LAUNCHER=<ON, OFF>                                   \
      -DTensorFlow_DIR=<path-to-tensorflow> ..
make -j$(nproc)
```

## Threading
For thread settings please refer to [Threading](../docs/DLBench_nthreads.md).

OpenMP settings are applicable for TF launcher. Additionally TF launcher uses `inter op parallelism threads` == 2 and  `intra op parallelism threads` == `nthreads` parameter value (if set)
