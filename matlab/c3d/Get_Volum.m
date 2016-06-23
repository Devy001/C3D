function volume =  Get_Volum(Dir, start, length, new_height, new_width) 
  %Dir = '/home/dongxuanyi/data/ucf101_rgb_img/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01';
  image = imread(sprintf('%s/image_%04d.jpg',Dir,start));
  rgbs = zeros(new_height, new_width, 3, length, 'single');
  
  for index = 0:length-1
    image = imread(sprintf('%s/image_%04d.jpg',Dir,start+index));
    image = MakeRGB(image, new_height, new_width);
    rgbs(:,:,:,index+1) = image;
  end

  volume = permute(rgbs(:,:,[3,2,1],:),[2,1,4,3]);

end


function image = MakeRGB(image, new_height, new_width)
  if (size(image,3) == 1)
    image = cat(3,image,image,image);
  end
  image = imresize(image, [new_height, new_width]);
end
