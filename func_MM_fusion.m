function  y_F_D_out = func_MM_fusion(imi,imv,Psi_x,Psi_cx,Psi_y,Psi_cy,D_joint)

blocksize = 8;
D = normcols(D_joint); %把每列的二范数变为1
% blocksize %
if (numel(blocksize)==1)
  blocksize = ones(1,2)*blocksize;
end
% maxatoms %
  maxatoms = 8;
% stepsize %
  stepsize = ones(1,2);
epsilon = 0.001;   % target error for omp

% denoise the signal %
G = D'*D;  
nz = 0;  % count non-zeros in block representations
beat=1;

% indices of the current block
ids1 = 1:blocksize(1);
ids2 = 1:blocksize(2);
x1 = imi;
x2 = imv;
blocks1 = im2col(x1(:,ids2),blocksize,'sliding');
blocks1 = blocks1(:,1:stepsize(1):end);
blocks2 = im2col(x2(:,ids2),blocksize,'sliding');
blocks2 = blocks2(:,1:stepsize(1):end);
tao = [];
taog = [];
taozhang=[];
[blocks1_dc, dc_1] = remove_dc(blocks1,'columns');
[blocks2_dc, dc_2] = remove_dc(blocks2,'columns');
blocks = [blocks1_dc;blocks2_dc];
y_F_D = zeros(size(x2));
gamma = omp2(D'*blocks,sum(blocks.*blocks),G,epsilon,'maxatoms',maxatoms);
for i=1:size(blocks1,2)
    taovaluegao= 1/(1+exp(((-beat)*(norm(blocks1_dc(:,i)/1,2) - norm(blocks2_dc(:,i)/1,2)))));
    taovalue= 1/(1+exp(((-beat)*(norm(dc_1(:,i)/1,2) - norm(dc_2(:,i)/1,2)))));
    taovaluezhang= 1/(1+exp(((-beat)*(norm(gamma(257:512,:),1) - norm(gamma(513:768,:),1)))));
%     taovaluezhang= norm(gamma(257:512,:),1)/(norm(gamma(257:512,:),1)+norm(gamma(513:768,:),1));
%     if norm(gamma(257:512,:),1) ==0 && norm(gamma(513:768,:),1)==0
%         taovaluezhang = 0.5;
%     end
    taog = [taog taovaluegao];
    tao = [tao taovalue];
    taozhang = [taozhang taovaluezhang];
end
    nz = nz + nnz(gamma);
    blocks_F_D=  (ones(64,1)* taozhang).* (Psi_cx*gamma(1:256,:)) + (ones(64,1)* (1-taozhang)).* (Psi_cy*gamma(1:256,:))+(ones(64,1)* taog).*(Psi_x*gamma(257:512,:)) + .......
    + (ones(64,1)* (1-taog)).*(Psi_y*gamma(513:768,:)) + ones(64,1)*( tao.* dc_1) + ones(64,1) * ((1-tao).*dc_2);
    k = 1;   % the index of the current block in the blocks matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  sum the cleaned blocks into y
%  every time the current batch of signals is exhausted, extract the next
%  batch of signals and denoise them
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lastids contains the indices of the last block in each dimension
    lastids = stepsize .* floor((size(x1)-blocksize)./stepsize) + 1;
    blocknum = prod(floor((size(x1)-blocksize)./stepsize) + 1);
    tid = timerinit('omprec', blocknum); lastmsgcheck = 0;
for i = 1 : blocknum
   y_F_D(ids1,ids2) = y_F_D(ids1,ids2) + reshape(blocks_F_D(:,k),blocksize);
  if (i<blocknum) 
    if (ids1(1) == lastids(1))
        tao = [];
        taog = [];
        taozhang=[];
        ids1 = 1:blocksize(1);
        ids2 = ids2+stepsize(2);
        blocks1 = im2col(x1(:,ids2),blocksize,'sliding');
        blocks1 = blocks1(:,1:stepsize(1):end);
        blocks2 = im2col(x2(:,ids2),blocksize,'sliding');
        blocks2 = blocks2(:,1:stepsize(1):end);
        [blocks1_dc, dc_1] = remove_dc(blocks1,'columns');
        [blocks2_dc, dc_2] = remove_dc(blocks2,'columns');
        blocks = [blocks1_dc;blocks2_dc];
        gamma = omp2(D'*blocks,sum(blocks.*blocks),G,epsilon,'maxatoms',maxatoms);
        %%%%%选择融合规则

        for j=1:size(blocks1,2)
            taovaluegao= 1/(1+exp(((-beat)*(norm(blocks1_dc(:,j)/1,2) - norm(blocks2_dc(:,j)/1,2)))));
            taovalue= 1/(1+exp(((-beat)*(norm(dc_1(:,j)/1,2) - norm(dc_2(:,j)/1,2)))));
              taovaluezhang= 1/(1+exp(((-beat)*(norm(gamma(257:512,:),1) - norm(gamma(513:768,:),1)))));
%             taovaluezhang= norm(gamma(257:512,:),1)/(norm(gamma(257:512,:),1)+norm(gamma(513:768,:),1));
%                 if norm(gamma(257:512,:),1) ==0 && norm(gamma(513:768,:),1)==0
%                     taovaluezhang = 0.5;
%                end
            taog = [taog taovaluegao];
            tao = [tao taovalue];
            taozhang = [taozhang taovaluezhang];
        end
            nz = nz + nnz(gamma);
    blocks_F_D=  (ones(64,1)* taozhang).* (Psi_cx*gamma(1:256,:)) + (ones(64,1)* (1-taozhang)).* (Psi_cy*gamma(1:256,:))+(ones(64,1)* taog).*(Psi_x*gamma(257:512,:)) + .......
    + (ones(64,1)* (1-taog)).*(Psi_y*gamma(513:768,:)) + ones(64,1)*( tao.* dc_1) + ones(64,1) * ((1-tao).*dc_2);
          k = 1;
    else
        ids1 = ids1+stepsize(1);
        k = k+1;
    end
  end

end

cnt1 = countcover(size(x1),blocksize,stepsize);
y_F_D_out = round(y_F_D./cnt1);
