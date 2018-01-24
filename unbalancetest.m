%clc;
%clear;
%load('LR.mat')
function [Cm,value3,groupno,avg2,tresult2,position2] = unbalancetest(Data,Data2,k,mini,maxi)
%Data为原始数据，Data2为测试数据，k为调整后比例为1：k,tresult2为修正后最优错误率
tic
[xd,yd] = size(Data{1,1});
for i = 1:5
    result{1,i} = ones(4,16);
    result2{1,i} = ones(4,16);
end
for i = mini:maxi
    i;
    if k == 2
        ncount = 51;
    end
    if k == 3
        ncount = 30;
    end
    if k == 4
        ncount = 13;
    end
    if k == 5
        ncount = 1;
    end
    X(1:120,1:yd)=Data{i,1};   %1~120 target
    X(121:720,1:yd)=Data{i,2};  %121~600 nontarget
    tag=[repmat((1),120,1);repmat((0),600,1)];
    newX = X;
    newtag = tag;
    %%
    %ODR  reduce sample
    dnont = zeros(600,600); %distance matrix
    for pnont = 1:600
        for qnont = (pnont+1):600
            dnont(pnont,qnont)=sum(X(pnont+120,:)-X(qnont+120,:)).^2;
        end
    end
    dnont = dnont+dnont';
    dnont = [dnont,zeros(600,120)];   %distance matrix nontarget(row,600) to [nontarget + target](col,600+120)
    for pnont = 1:600
        for qt = 1:120
            dnont(pnont,qt+600) = sum(X(pnont+120,:)-X(qt,:)).^2;
        end
    end
    count = 0;
    for flag = 1:600
        t = sort(dnont(flag,:));
        col1 = find(dnont(flag,:)<=t(6),6);
        col11 = find(dnont(flag,col1)>t(1),5);
        col1 = col1(col11);
        withp = sum(col1<=600);  %number classified by KNN with p
        col2 = find(dnont(flag,:)<=t(7),7);
        col22 = find(dnont(flag,col2)>t(1),6);
        col2 = col2(col22);        
        withoutp = sum(col2<=600); %number classified by KNN without p
        if withp < withoutp    %p has litte influence on the classifier
            newX(flag+120,:)=[];
            newtag(flag+120,:)=[];
            count = count+1;
        end
        if count >= ncount*5
            break;
        end
    end


    %%
    
    %BSMOTE - add sample
    adding = zeros(ncount,yd);
    dt = dnont(:,601:720)';
    tar2tar = zeros(120,120);
    for pt = 1:120
        for qt = (pt+1):120
            tar2tar(pt,qt) = sum(X(pt,:)-X(qt,:)).^2;
        end
    end
    tar2tar = tar2tar+tar2tar';
    dt = [dt,tar2tar]; %distance matrix target(row,120) to [nontarget + target](col,600+120)
    B=[];
    knn = [];
    for flag = 1:120
        t = sort(dt(flag,:));
        col1 = find(dt(flag,:)<=t(6),6);
        col11 = find(dt(flag,col1)>t(1),5);
        col1 = col1(col11);  %column location 0~720 [nontarget + target]
        if (sum(col1>600)>0)&&(sum(col1<=600)>0)   %boundary sample set
            B = [B;flag]; %row location (home, target)
        end
    end
    for bb = 1:length(B)
        flag = B(bb);
        t = sort(dt(flag,601:720));
        t = t(2:end);
        col = find(dt(flag,601:720)<=t(5),5);   %column location 1~120 [target]
        knn = [knn;col];
    end
    flag = zeros(ncount,1);         %row position of chosen neighbor 1·120 [t]
    cr = 1+fix(rand(1,ncount)*length(knn(1,:)));
    rr = 1+fix(rand(1,ncount)*length(knn(:,1)));
    for pos = 1:ncount
        flag(pos,1) = knn(rr(pos),cr(pos));
        near(pos,:) = X(knn(rr(pos),cr(pos)),:);
        home(pos,:) = X(B(rr(pos)),:);
    end
    dif = near - home;
    for pos = 1:ncount
        t = sort(dt(flag(pos),:));
        t = t(2:end);
        col = find(dt(pos,:)<=t(5),5);
        if (sum(col>600)>0)&&(sum(col<=600)>0)
            randd = rand(1,1);
        else randd = rand(1,1)/2;
        end
        adding(pos,:)=home(pos,:)+randd*dif(pos);
    end
    newX = [adding;newX];
    newtag = [ones(ncount,1);newtag];

%%
%CV
    X = newX;
    tag = newtag;
    Y = ismember(tag,1);
    P = cvpartition(Y,'Holdout',0.20);
    training = X(P.training,:);   %training+validation
    trtarget = Y(P.training);
    trainingset{i,1} = X(P.training,:);
    trainingset{i,2} = Y(P.training);
    testing{i,1} = X(P.test,:);        %testing
    testing{i,2} = Y(P.test);
    indices = crossvalind('Kfold',length(training),5);
    for j = 1:5
        % Use a linear support vector machine classifier
        validate = (indices == j);
        train = ~validate;
        trainin = training(train, :);
        traintar = trtarget(train, :);
        validatein = training(validate, :);
        validatetar = trtarget(validate, :);
        
%         svmStruct = svmtrain(trainin,traintar,'kernel_function','quadratic');
%         C1 = svmclassify(svmStruct,validatein);
%         result{1,j}(1,i) = sum(validatetar~= C1)/length(validatetar);  %mis-classification rate
%         C = svmclassify(svmStruct,X(P.test,:),'showplot',true);
%         conMat = confusionmat(Y(P.test),C); % the confusion matrix
%         errRate2 = perr(conMat);
%         result2{1,j}(1,i) = errRate2;
        svmStruct = svmtrain(trainin,traintar,'kernel_function','rbf');
        C2 = svmclassify(svmStruct,validatein);
        result{1,j}(2,i) = sum(validatetar~= C2)/length(validatetar);
        C = svmclassify(svmStruct,X(P.test,:),'showplot',true);
        conMat = confusionmat(Y(P.test),C); % the confusion matrix
        errRate2 = perr(conMat);
        result2{1,j}(2,i) = errRate2;
        svmStruct = svmtrain(trainin,traintar,'kernel_function','polynomial');
        C3 = svmclassify(svmStruct,validatein);
        result{1,j}(3,i) = sum(validatetar~= C3)/length(validatetar);  %mis-classification rate
        C = svmclassify(svmStruct,X(P.test,:),'showplot',true);
        conMat = confusionmat(Y(P.test),C); % the confusion matrix
        errRate2 = perr(conMat);
        result2{1,j}(3,i) = errRate2;
        svmStruct = svmtrain(trainin,traintar,'kernel_function','mlp');
        C4 = svmclassify(svmStruct,validatein);
        result{1,j}(4,i) = sum(validatetar~= C4)/length(validatetar);
        C = svmclassify(svmStruct,X(P.test,:),'showplot',true);
        conMat = confusionmat(Y(P.test),C); % the confusion matrix
        errRate2 = perr(conMat);
        result2{1,j}(4,i) = errRate2;
    end
end
avg =  sum(cat(3, result{:}),3)./5;%the average of 5 validation results
[value,position] = min(avg);
avg2 =  sum(cat(3, result2{:}),3)./5;%the average of 5 validation results
[value2,position2] = min(avg2);
tresult = zeros(1,maxi);
tresult2 = zeros(1,maxi);
for m = 1:maxi
    m
    svmStruct = svmtrain(trainingset{m,1},trainingset{m,2},'kernel_function',kernelname(position(m)));
    C = svmclassify(svmStruct,testing{m,1});
    err = errmat(testing{m,2},C);
    %tresult(1,m) = sum(testing{m,2}~= C)/length(testing{m,2});  %mis-classification rate
    tresult(1,m) = perr(err);
end
for m = 1:maxi
    m
    svmStruct = svmtrain(trainingset{m,1},trainingset{m,2},'kernel_function',kernelname(position2(m)));
    C = svmclassify(svmStruct,testing{m,1});
    err = errmat(testing{m,2},C);
    %tresult2(1,m) = sum(testing{m,2}~= C)/length(testing{m,2});  %mis-classification rate
    tresult2(1,m) = perr(err);
end
[xd2,yd2] = size(Data2);
[value3,groupno] = min(tresult2);
Cm = zeros(1,xd2);
m = groupno;
svmStruct = svmtrain(trainingset{m,1},trainingset{m,2},'kernel_function',kernelname(position2(m)));
C = svmclassify(svmStruct,Data2);
Cm(1,:) = C(:,1);
toc