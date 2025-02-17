% sequence generation
%% Step 1: Generate 100,000 Random Sequences
close all; clear; clc;
disp('Generating Random Sequences...')
tic
rng('shuffle');  % Use current time as random seed
L = 120;  % Sequence length corresponding to 1 second stimulus at 120 Hz
numSequences = 100000; 

original_sequence = 2 * rand(numSequences, L) - 1; % Generate sequences in the range [-1, 1]
save('original_sequence.mat', 'original_sequence');
toc

%% Step 2: Filter the Sequences (15-25 Hz)
clear; clc;
disp('Filtering Sequences...')
tic
load("original_sequence.mat");
fs = 120;  % Sampling rate of 120 Hz

% Design 15-25 Hz bandpass filter
d2 = designfilt('bandpassiir', ...       % Filter type
    'StopbandFrequency1', 8, ...    % Frequency constraints
    'PassbandFrequency1', 15, ...
    'PassbandFrequency2', 25, ...
    'StopbandFrequency2', 32, ...
    'StopbandAttenuation1', 40, ...  % Attenuation constraints
    'PassbandRipple', 0.1, ...
    'StopbandAttenuation2', 40, ...
    'DesignMethod', 'ellip', ...      % Design method
    'MatchExactly', 'passband', ...
    'samplerate', fs);           % Sampling rate
% Handle filter boundary issue: Extend sequence and filter
extended_sequence = repmat(original_sequence, 1, 3);  % Repeat 3 times
filtered_extended = filtfilt(d2, extended_sequence')'; % Zero-phase filtering
filtered_sequence = filtered_extended(:, L+1:2*L); % Take the middle 120 points

% Normalize to [-1, 1]
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
    x = x - mean(x);  % Remove mean
    energy(i) = sum(x.^2); % Calculate energy
    % Envelope normalization
    analytic_signal = hilbert(x);
    envelope = abs(analytic_signal);
    chosen_sequences(i, :) = x ./ envelope; % Normalize
end

% Sort by energy in descending order and select the top 40,000
selected_sequences = chosen_sequences;
save('chosen_sequence.mat', 'selected_sequences');
toc

%% Step 4: Greedy Selection of Low-Correlation Sequences
close
clear
disp('Selecting Low-Correlation Sequences')
tic
load chosen_sequence.mat
nTargets = 240;
% Frequency Band 1
X1 = chosen_sequence;
iY = X1;
totalnumber = size(iY, 1);
co1 = corrcoef(X1');

co1 = abs(co1) + 100 * eye(totalnumber);
k = 1; % k > 1 used for beam search
index = zeros(nTargets, k, nTargets);

sumline = sum(co1);
[Ie, De] = sort(sumline);
index(1, :, 1) = De(1:k);
for N = 2:nTargets

    corrmatrix = zeros(k, totalnumber, N);

    for i = 1:N-1 % Calculate the sum of correlation coefficients
        corrmatrix(:,:,i) = co1(index(i, :, N-1), :);
        corrmatrixc = max(corrmatrix, [], 3);
    end

    [Ie, De] = sort(reshape(corrmatrixc, 1, k*totalnumber));

    for j = 1:k
        row = mod(De(j), k);
        if row == 0
            row = row + k;
        end
        column = ceil(De(j) / k);
        index(1:N-1, j, N) = index(1:N-1, row, N-1);
        index(N, j, N) = column;
    end
end
championsequence = iY(index(1:nTargets, 1, nTargets), :);
save('champion_sequencetest1.mat', 'championsequence');
toc
