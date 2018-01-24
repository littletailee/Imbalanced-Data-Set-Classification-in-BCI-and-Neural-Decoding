function [label, score, accuracy, precision, recall, F] = SVMensemble(train,traintarget,test,testtar,kernel,ensemble)
%train:training set
%traintarget:training lable
%test:testing set
%kernel:vector of kernels
%ensemble:number of ensembles
%train = training;
%traintarget = trtarget;
%test = testing;
%testtar = testingtar;
%kernel = {'RBF';'polynomial';'linear'};
%ensemble = 3;

[xd,~] = size(train);    
[yd,~] = size(test);
labels = zeros(yd,ensemble);
scores = zeros(yd,ensemble);
    if ensemble > 1
        for i = 1:ensemble
            pos = ceil(rand(xd,1)*xd);
            training = train(pos,:);
            trtarget = traintarget(pos,:);  
            svmStruct = fitcsvm(training,trtarget,'KernelFunction',kernel{i},'BoxConstraint',10000);
            [C2,score] = predict(svmStruct,test);
            labels(:,i) = (score(:,2) > quantile(score(:,2),0.8333));
 %           labels(:,i) = C2;
            scores(:,i) = score(:,2);
        end
        label = round(sum(labels,2)/ensemble, 0);
        score = mean(scores,2);
    else
        svmStruct = fitcsvm(train,traintarget,'KernelFunction',kernel,'BoxConstraint',10000);
        [C2,score] = predict(svmStruct,test);
        label = (score(:,2) > quantile(score(:,2),0.86));
%        label = C2;
        score = score(:,2);
    end
    accuracy = sum(label==testtar)/yd;
    tp = sum(label==1 & testtar==1);
    fp = sum(testtar==0 & label == 1);
    fn = sum(testtar==1 & label == 0);
    precision = tp/(tp + fp);
    recall = tp/(tp + fn);
    F = 2 * precision * recall / (precision + recall);
end