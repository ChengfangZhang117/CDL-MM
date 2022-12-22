clear all
clc
addpath('couple_dictionary_learning_codes')
addpath(genpath('normtool'));
addpath(genpath('ksvdbox'));
addpath(genpath('ompbox'));
addpath(genpath('private'));
addpath(genpath('fusion_evaluation'))
addpath('load_imagetool');
addpath('./Dicts');


load('CDL_T1T2_D64x256_WT0.5.mat');
clear paramsCDL
O=zeros(64,256);
D_joint=[outputCDL.Psi_cx outputCDL.Psi_x O;outputCDL.Psi_cy O outputCDL.Psi_y];
%% Fusion
method = {'CDL_song'};
params.method = method;
Evaluation_index = {'QS','ƽ���ݶ�','��Եǿ��','��׼��','�ҶȾ�ֵ','ͼ��������','QG','Qabf','Q0','Qw','Qe','NMI','QTE','QNCIE','QM','QSF','QP','QC','QY','QCV','QCB','psnr','rmse','�ռ�Ƶ��','ssim','viff'};
test_image=mygetdirfiles('test_images');
test_imagecell=load_image(test_image);
index_a = 1:2:size(test_imagecell);
index_b = 2:2:size(test_imagecell);
A=cell(size(index_a,2),1);
B=cell(size(index_b,2),1);
k = 1;%%��ʾ����
t = 1;%%���Ʒ�������
q = 1;
m = 1;%%sheet
for i = 1:numel(test_imagecell)/2
    f = test_image{index_a(i)};
    [p, n, x] = fileparts(f);
    params.p = p;
    params.n = n;
    params.x1 = x;
    xlswrite('EvalResult.xls',{n},m,['A',num2str(t)]);
    xlswrite('EvalResult.xls',Evaluation_index,m,['B',num2str(t)]);
    xlswrite('EvalResult.xls',method',m,['A',num2str(t+1)]);
    xlswrite('time.xls',{n},m,['A',num2str(t)]);
    xlswrite('time.xls',method',m,['A',num2str(t+1)]);
     if size(test_imagecell{index_a(i)},3)>1
    A{i}=double(rgb2gray(test_imagecell{index_a(i)}));
    B{i}=double(rgb2gray(test_imagecell{index_b(i)}));
    else
    A{i}=double(test_imagecell{index_a(i)});
    B{i}=double(test_imagecell{index_b(i)});
     end
    tic;
    y_F_CDL_song{i}=func_MM_fusion(A{i},B{i},outputCDL.Psi_x,outputCDL.Psi_cx,outputCDL.Psi_y,outputCDL.Psi_cy,D_joint);
     time_CDL_song=toc;
    EvalResult_CDL_song = Evaluation(A{i},B{i},double(y_F_CDL_song{i}),256);
    xlswrite('EvalResult.xls', EvalResult_CDL_song,m,['B',num2str(t+1)]);
    xlswrite('time.xls', time_CDL_song,m,['B',num2str(t+1)]);
      
    t = t + size(method,2) + 1;
    q = q + 3;
    k = 1;
%%%�����ַ������õ����ں�ͼ��ŵ�ͬһ����ľ�����
   result = cat(3,y_F_CDL_song{i});
   conf.fusion_image{i} = {};
%%%%���ں�ͼ��д�뵽ָ�����ļ�����
   for j = 1:numel(method)
        conf.fusion_image{i}{j} = fullfile(p, 'results', [n sprintf('[%d-%s]', j, method{j}) x]);
        imwrite(uint8(result(:, :, j)), conf.fusion_image{i}{j});%%%%�����ַ����Ľ���ŵ�result�ļ�����
    end
end


