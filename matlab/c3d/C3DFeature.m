%function C3DFeature()

clear;clc;
current_dir = pwd;
caffe_dir = '../../'; cd(caffe_dir); caffe_dir = pwd;
cd(current_dir);
addpath(fullfile(caffe_dir,'matlab','caffe'));


caffe('reset');
proto = fullfile(caffe_dir,'examples','C3D','c3d_devlope.proto');
weight = fullfile(caffe_dir,'examples','C3D','c3d_ucf101_final.caffemodel');
gpu_id = 0;

crop_size = 112;
new_height = 128;
new_width = 171;
new_length = 16;
h_off = uint8((new_height - crop_size) / 2);
w_off = uint8((new_width - crop_size) / 2);
mean_file = fullfile(caffe_dir,'examples','C3D','train01_16_128_171_mean.binaryproto');

if caffe('is_initialized') == 0
  if exist(weight, 'file') == 0
    % NOTE: you'll have to get the pre-trained ILSVRC network
    error('You need a network model file');
  end
  if ~exist(proto,'file')
    % NOTE: you'll have to get network definition
    error('You need the network prototxt definition');
  end
  caffe('init', proto, weight)
end

if gpu_id >= 0
  fprintf('Using GPU Mode\n');
  caffe('set_mode_gpu');
else
  fprintf('Using CPU Mode\n');
  caffe('set_mode_cpu');
  caffe('set_device', gpu_id);
end
% put into test mode
caffe('set_phase_test');
fprintf('Done with set_phase_test\n');

video_mean = caffe('get_meanfile', mean_file);
fprintf('Load video mean file done\n');

%{
fprintf('Start Testing\n');
[video_path, start, label] = textread(fullfile(caffe_dir,'examples','C3D','ucf_test.lst'), '%s%d%d');
total = length(video_path);
right = 0;
for index = 1:total
  tic;
  Dir = fullfile('/home/dongxuanyi/data/ucf101_rgb_img/', video_path{index});
  volume =  Get_Volum(Dir, start(index), new_length, new_height, new_width);
  volume = volume - video_mean;
  input_data = {volume(w_off:w_off+crop_size-1, h_off:h_off+crop_size-1, :, :)};
  scores = caffe('forward', input_data);
  scores = reshape(scores{1}, numel(scores{1}), 1);
  [value, idx] = max(scores);
  if idx == label(index)+1
    right = right + 1;
  end
  %fc6 = caffe('get_features', 'fc6');
  fprintf('%4d / %4d, Ave accuracy : %.4f, cost : %.2f s\n', index, total, right / index, toc); 
end
%}


DIR = '/home/dongxuanyi/Extract/TrainValVideo_Img';
video_list = dir(fullfile(DIR, 'video*'));
fc6 = cell(length(video_list), 1);
fc7 = cell(length(video_list), 1);
fprintf('%s has %5d videos\n', DIR, length(video_list));
feature_dim = 4096;
for i = 1:length(video_list)
  tic;
  video = fullfile(DIR, video_list(i).name);
  images = dir(fullfile(video, 'image*'));
  cur_total = length(images);
  if (cur_total < new_length) 
    fprintf('%4d : %s less than 16 frames\n', i, video);
    continue;
  end

  L = floor(cur_total / new_length);
  cur_fc6 = zeros(L, feature_dim, 'single');
  cur_fc7 = zeros(L, feature_dim, 'single');
  for index = 1:L
    volume =  Get_Volum(video, (index-1)*16, new_length, new_height, new_width);
    volume = volume - video_mean;
    input_data = {volume(w_off:w_off+crop_size-1, h_off:h_off+crop_size-1, :, :)};
    scores = caffe('forward', input_data);
    cfc6 = caffe('get_features', 'fc6');
    cfc7 = caffe('get_features', 'fc7');
    cur_fc6(index,:) = reshape(cfc6, feature_dim, 1);
    cur_fc7(index,:) = reshape(cfc7, feature_dim, 1);
  end
  fc6{i} = cur_fc6;
  fc7{i} = cur_fc7;
  fprintf('%04d / %-4d : %-8s , cost %.3f s\n', i, length(video_list), video_list(i).name, toc);
end

save('/home/dongxuanyi/C3D_Trainval_Feature.mat', 'video_list', 'fc6', 'fc7', '-v7.3');
