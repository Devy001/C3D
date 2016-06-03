clear;clc;
current_dir = pwd;
caffe_dir = '../../'; cd(caffe_dir); caffe_dir = pwd;
cd(current_dir);
addpath(fullfile(caffe_dir,'matlab'));
caffe.reset_all();

proto = '/home/dongxuanyi/code/OK/caffe-binary/examples/c3d_finetuning/c3d_test.proto';
net = caffe.Net(proto, 'test');

