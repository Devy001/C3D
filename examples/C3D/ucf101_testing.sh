if [ ! -n "$1" ] ;then
    echo "$1 is empty, default is 0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi

GLOG_logtostderr=1 ../../build/tools/test_net.bin \
    c3d_test.proto \
    snapshot/c3d_ucf101_finetune_whole_iter_5000 836 GPU 6
#    c3d_ucf101_final.caffemodel 836 GPU $gpu
