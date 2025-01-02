clear 
subname={'j','t','gyx','hlt','rj','yxw','wt','xl','ljm','h','gxx','gm','zxj','tjc','p','m','sna' ,'hlj'};%'j','t','gyx','hlt','rj','yxw','wt','xl','ljm','h','gxx','gm','zxj','tjc','p','m','sna' ,'hlj'
for sub=1:length(subname)
load([subname{sub},'vmcresult.mat']);
load('champion_sequencetest.mat')
for type1=1:240
    y1=championsequence(type1,:);
    y2=downsample(reshape(repmat(y1,25,1),1,[]),3);
    y2=mapminmax(y2,0,1);     

    test_datay(type1,:)=y2;   
end
numTests = 240;
bestAccuracy = 0;


 correctClassifications=0;

    for test = 1:numTests
       
        testSequence = test_datay(:, 1:1000);
        typeSequences =preddata1(test, 1:1000, 1);
        ccc = abs(corr(typeSequences', testSequence'));
        [~, result] = max(ccc);
        if test == result
            correctClassifications = correctClassifications + 1;
        end
    end

    % Calculate accuracy for the current delay
    accuracy = correctClassifications / numTests;

    % Update best delay if the average correlation or accuracy is higher

    accuracys(sub)=accuracy;
end






