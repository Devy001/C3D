%{
for index = 1:3
[train{index}, label_train{index}] = textread(['trainlist0' int2str(index) '.txt'],'%s %d');
[test{index}] = textread(['testlist0' int2str(index) '.txt'],'%s');
end

label_test = cell(1,3);
for index = 1:3
   tic;
   label_test{index} = zeros(length(test{index}),1);
   for jj = 1:length(test{index})
      for ii = 1:3
          if(ii == index) continue; end
          for k = 1:length(train{ii})
             if(strcmp( train{ii}(k), test{index}(jj)) == 1)
                 label_test{index}(jj) = label_train{ii}(k);
                 break;
             end
          end
      end
   end
   fprintf('Handle %5d pictures cost %f s\n',length(test{index}),toc);
end

save('ucf_all.mat','train','test','label_train','label_test','-v7.3');
%}
%%% Generate Train Data
split = 1;
data = train{split};
label = label_train{split};
root = '/home/dongxuanyi/data/ucf101_rgb_img';
fid = fopen('ucf_train.lst','w');
L = 16;
tic;
for index = 1:length(data)
    name = data{index}(1:end-4);
    filelist = dir(fullfile(root, name, '*.jpg'));
    for A = 0 : L : length(filelist)-L
        fprintf(fid, '%s %d %d\n', name, A, label(index));
    end
end
fclose(fid);
fprintf('Generate Train Data Cost %f s\n', toc);

%%% Generate Test Data
data = test{split};
label = label_test{split};
root = '/home/dongxuanyi/data/ucf101_rgb_img';
fid = fopen('ucf_test_name_total_label.lst','w');
L = 16;
for index = 1:length(data)
    name = data{index}(1:end-4);
    filelist = dir(fullfile(root, name, '*.jpg'));
    fprintf(fid, '%s %d %d\n', name, length(filelist), label(index));
end
fclose(fid);

