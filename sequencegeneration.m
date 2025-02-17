%sequencegeneration
%% Step 1: Generate 100,000 Random Sequences
close all; clear; clc;
disp('Generating Random Sequences...')
tic
rng('shuffle');  % 使用当前时间随机种子
L = 120;  % 序列长度对应 120 Hz 的 1 秒刺激
numSequences = 100000; 

original_sequence = 2 * rand(numSequences, L) - 1; % 生成 [-1, 1] 范围的序列
save('original_sequence.mat', 'original_sequence');
toc

%% Step 2: Filter the Sequences (15-25 Hz)
clear; clc;
disp('Filtering Sequences...')
tic
load("original_sequence.mat");
fs = 120;  % 采样率 120 Hz

% 设计 15-25 Hz 带通滤波器
d2= designfilt('bandpassiir', ...       % 响应类型
    'StopbandFrequency1',8, ...    % 频率约束
    'PassbandFrequency1',15, ...
    'PassbandFrequency2',25, ...
    'StopbandFrequency2',32, ...
    'StopbandAttenuation1',40, ...  % 幅度约束
    'PassbandRipple',0.1, ...
    'StopbandAttenuation2',40, ...
    'DesignMethod','ellip', ...      % 设计方法
    'MatchExactly','passband'   ,...
    'samplerate',fs);           % 采样率
% 处理滤波边界问题：扩展序列并进行滤波
extended_sequence = repmat(original_sequence, 1, 3);  % 复制 3 次
filtered_extended = filtfilt(d2, extended_sequence')'; % 零相位滤波
filtered_sequence = filtered_extended(:, L+1:2*L); % 取中间 120 位

% 归一化至 [-1,1]
normalized_sequence = mapminmax(filtered_sequence', -1, 1)'; 
save('Filtered_random_sequence.mat', 'normalized_sequence');
toc

%% Step 3: Envelope Normalization and Selection
clear; clc;
disp('Normalizing and Selecting Sequences...')
tic
load('Filtered_random_sequence.mat');

chosen_sequences = zeros(100000, L);
energy = zeros(size(normalized_sequence, 1), 1);

for i = 1:size(normalized_sequence, 1)
    x = normalized_sequence(i, :);
    x = x - mean(x);  % 去均值
    energy(i) = sum(x.^2); % 计算能量
    % 包络归一化
    analytic_signal = hilbert(x);
    envelope = abs(analytic_signal);
    chosen_sequences(i, :) = x ./ envelope; % 归一化
end

% 依据能量降序排序并选择前 40000 个
selected_sequences = chosen_sequences;
save('chosen_sequence.mat', 'selected_sequences');
toc

%% Step 4: Greedy Selection of Low-Correlation Sequences
close
clear
disp('筛选相关')
tic
load chosen_sequence.mat
nTargets = 240;
% 频段1
X1=chosensequence;
iY=X1;
totalnumber=size(iY,1);
co1=corrcoef(X1');

co1=abs(co1)+100*eye(totalnumber);
k=1 ;%k>1时作为beamsearch
index=zeros(nTargets,k,nTargets);

  
sumline=sum(co1);
[Ie,De]=sort(sumline);
index(1,:,1)=De(1:k);
 for N=2:nTargets

corrmatrix=zeros(k,totalnumber,N);

    for i=1:N-1%计算相关系数的和
   corrmatrix(:,:,i)=co1(index(i,:,N-1),:);
   corrmatrixc=max(corrmatrix,[],3);
    end

   [Ie,De]=sort(reshape(corrmatrixc,1,k*totalnumber));
  
     for j=1:k
        row=mod(De(j),k);
        if row==0
            row=row+k;
        end
        column=ceil(De(j)/k);
        index(1:N-1,j,N)=index(1:N-1,row,N-1);
        index(N,j,N)=column;
     end
 end
championsequence=iY(index(1:nTargets,1,nTargets),:);
save('champion_sequencetest1.mat','championsequence');
toc
 
