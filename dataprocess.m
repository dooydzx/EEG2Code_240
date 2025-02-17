% Clear workspace
clear

% Define subject names
subname ={'j','t','gyx','hlt','rj','yxw','wt','xl','ljm','h','gxx','gm','zxj','tjc','p','m','sna' ,'hlj'};

% Load training and testing data
load("allSubjects_testonedata.mat")
load('allSubjects_trainonedata.mat')

% Loop through each subject
for sub=1:length(subname)

    % Define filter parameters
    fs = 1000.0;  % Sampling frequency
    lowcut = 5.0;  % Low cutoff frequency
    highcut = 45.0; % High cutoff frequency
    nyq = 0.5 * fs; % Nyquist frequency
    low = lowcut/nyq;
    high = highcut/nyq;
    order = 4; % Filter order
    [b, a] = butter(order, [low, high], 'bandpass'); % Design Butterworth bandpass filter

    % Load stimulus sequences
    load('champion_sequencetrain.mat')
    
    % Process the training labels
    for type1=1:30
        y1 = championsequence(type1,:); % Get sequence for current type
        y2 = downsample(reshape(repmat(y1,25,1),1,[]),3); % Expand and downsample
        y2 = mapminmax(y2,0,1); % Normalize to range [0,1]
        y2 = round(y2); % Convert to binary values
        train_datay(type1,1:1000) = y2(1:1000);
    end
    
    % Duplicate training labels for trials
    for type1=1:30
        for trial=1:12
            train_data_y((trial-1)*30+type1,:) = train_datay(type1,:);
        end
    end
    
    % Preprocess training EEG data
    for type=1:360
        for ch=1:21
            filted(type,ch,:) = filtfilt(b,a,squeeze(train_data.(subname{sub})(type,ch,:))); % Apply bandpass filter
            train_data_x(type,ch,:) = mapminmax(filted(type,ch,1040:2039),-1,1); % Normalize data
        end
    end
    
    % Preprocess test EEG data
    for type=1:240
        for ch=1:21
            filted2(type,ch,:) = filtfilt(b,a,squeeze(onetest.(subname{sub})(type,ch,:))); % Apply bandpass filter
            test_data_x(type,ch,:) = mapminmax(filted2(type,ch,1040:2210),-1,1); % Normalize data
        end
    end
    
    % Save processed data
    save([subname{sub},'v1.mat'],'train_data_x','train_data_y','test_data_x')
end

% Clear workspace again for the next dataset
clear

% Define subject names
subname ={'j','t','gyx','hlt','rj','yxw','wt','xl','ljm','h','gxx','gm','zxj','tjc','p','m','sna' ,'hlj'};

% Load training and multiple test datasets
load("allSubjects_testmultidata.mat")
load('allSubjects_trainonedata.mat')

% Loop through each subject
for sub=1:length(subname)

    % Define filter parameters
    fs = 1000.0;
    lowcut = 5.0;  
    highcut = 45.0;
    nyq = 0.5 * fs;
    low = lowcut/nyq;
    high = highcut/nyq;
    order = 4;
    [b, a] = butter(order, [low, high], 'bandpass');

    % Load stimulus sequences
    load('champion_sequencetrain.mat')
    
    % Process the training labels
    for type1=1:30
        y1 = championsequence(type1,:);
        y2 = downsample(reshape(repmat(y1,25,1),1,[]),3);
        y2 = mapminmax(y2,0,1);     
        y2 = round(y2);
        train_datay(type1,1:1000) = y2(1:1000);
    end
    
    % Duplicate training labels for trials
    for type1=1:30
        for trial=1:12
            train_data_y((trial-1)*30+type1,:) = train_datay(type1,:);
        end
    end
    
    % Preprocess training EEG data
    for type=1:360
        for ch=1:21
            filted(type,ch,:) = filtfilt(b,a,squeeze(train_data.(subname{sub})(type,ch,:)));
            train_data_x(type,ch,:) = mapminmax(filted(type,ch,1040:2039),-1,1);
        end
    end
    
    % Preprocess test EEG data
    for type=1:240
        for ch=1:21
            filted2(type,ch,:) = filtfilt(b,a,squeeze(multitest.(subname{sub})(type,ch,:)));
            test_data_x(type,ch,:) = mapminmax(filted2(type,ch,1040:2210),-1,1);
        end
    end
    
    % Save processed data
    save([subname{sub},'vm.mat'],'train_data_x','train_data_y','test_data_x')
end
