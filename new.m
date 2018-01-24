clear;
clc;
load('BFS.mat');
Datatrain = Data;
load('BFS_online.mat');
Datatest = data_online;
Cmr = zeros(4,720);
grno = zeros(1,4);
ervalue = zeros(1,3);
trainingset = [];9
testing = [];
tresult3 = zeros(4,16);
position = zeros(4,16);
avg = zeros(1,16);
for k = 1:16
    u = zeros(120,60);
    u(1,1:30) = (Datatrain{k,1}(120,1:30) + Datatrain{k,1}(1,1:30) + Datatrain{k,1}(2,1:30))/3;
    u(1,31:60) = (Datatrain{k,1}(120,31:60) + Datatrain{k,1}(1,31:60) + Datatrain{k,1}(2,31:60))/3;
    u(120,1:30) = (Datatrain{k,1}(120,1:30) + Datatrain{k,1}(119,1:30) + Datatrain{k,1}(1,1:30))/3;
    u(120,31:60) = (Datatrain{k,1}(120,31:60) + Datatrain{k,1}(119,31:60) + Datatrain{k,1}(1,31:60))/3;
    for r = 2:120 - 1
        u(r,1:30) = (Datatrain{k,1}(r - 1,1:30) + Datatrain{k,1}(r,1:30) + Datatrain{k,1}(r + 1,1:30))/3;
        u(r,31:60) = (Datatrain{k,1}(r - 1,31:60) + Datatrain{k,1}(r,31:60) + Datatrain{k,1}(r + 1,31:60))/3;
    end
    Datatrain{k,1}(1:120,1:60) = u(:,:);
    u = zeros(600,60);
    u(1,1:30) = (Datatrain{k,2}(600,1:30) + Datatrain{k,2}(1,1:30) + Datatrain{k,2}(2,1:30))/3;
    u(1,31:60) = (Datatrain{k,2}(600,31:60) + Datatrain{k,2}(1,31:60) + Datatrain{k,2}(2,31:60))/3;
    u(600,1:30) = (Datatrain{k,2}(600,1:30) + Datatrain{k,2}(599,1:30) + Datatrain{k,2}(1,1:30))/3;
    u(600,31:60) = (Datatrain{k,2}(600,31:60) + Datatrain{k,2}(599,31:60) + Datatrain{k,2}(1,31:60))/3;
    for r = 2:600 - 1
        u(r,1:30) = (Datatrain{k,2}(r - 1,1:30) + Datatrain{k,2}(r,1:30) + Datatrain{k,2}(r + 1,1:30))/3;
        u(r,31:60) = (Datatrain{k,2}(r - 1,31:60) + Datatrain{k,2}(r,31:60) + Datatrain{k,2}(r + 1,31:60))/3;
    end
    Datatrain{k,2}(1:600,1:60) = u(:,:);
end
for k = 1:16
    for r = 1:120
        Datatrain{k,1}(r,1:30) = guiyi(Datatrain{k,1}(r,1:30));%%%%%%%%%%%%%%修改处
        Datatrain{k,1}(r,31:60) = guiyi(Datatrain{k,1}(r,31:60));%%%%%%%%%%%%%%修改处
    end
    for r = 1:600
        Datatrain{k,2}(r,1:30) = guiyi(Datatrain{k,2}(r,1:30));%%%%%%%%%%%%%%修改处
        Datatrain{k,2}(r,31:60) = guiyi(Datatrain{k,2}(r,31:60));%%%%%%%%%%%%%%修改处
    end
end

Data = Datatrain;
[xd,yd] = size(Data{1,1});

%for i = 1:16
i=10;
    X(1:120,1:yd)=Data{i,1};   %1~120 target
    X(121:720,1:yd)=Data{i,2};  %121~600 nontarget
    tag=[repmat((1),120,1);repmat((0),600,1)];
    newX = X;
    newtag = tag;
%end

Datatest2 = zeros(720,60);
    for r = 1:5
        Datatest1 = zeros(144,60);
        for s = 1:r
            for u3 = 1:12
                for u1 = 1:12
                    for u2 = 1:30
                        Datatest1((u3 - 1)*12 + u1,u2) = Datatest1((u3 - 1)*12 + u1,u2) + Datatest{1,u3}(r,u1,u2);
                    end
                    for u2 = 31:60
                        Datatest1((u3 - 1)*12 + u1,u2) = Datatest1((u3 - 1)*12 + u1,u2) + Datatest{1,u3}(r,u1,u2);
                    end
                end
            end
        end
        Datatest1 = Datatest1/r;
        for s = 1:r
            for u3 = 1:12
                for u1 = 1:12
                    Datatest1((u3 - 1)*12 + u1,1:30) = guiyi(Datatest1((u3 - 1)*12 + u1,1:30));%%%%%%%%%%%%%%修改处
                    Datatest1((u3 - 1)*12 + u1,31:60) = guiyi(Datatest1((u3 - 1)*12 + u1,31:60));%%%%%%%%%%%%修改处
                end
            end
        end
        Datatest2((r - 1)*144 + 1:r*144,:) = Datatest1(:,:);
    end
    
    X = newX;
    tag = newtag;
    Y = ismember(tag,1);
    P = cvpartition(Y,'Holdout',0.20);
    training = X(P.training,:);   %training+validation
    trtarget = Y(P.training);
    testing = X(P.test,:);        %testing
    testingtar = Y(P.test);

    
kernels = {'RBF';'polynomial';'linear'};
AUCprereF = zeros(3,5);%AUC,accuracy,Precision, Recall,F
X1 = zeros(length(testingtar)+1,3);
Y1 = zeros(length(testingtar)+1,3);
scores = zeros(length(testingtar)+1,3);
for i = 1:length(kernels)
    [C2,scores, AUCprereF(i,2), AUCprereF(i,3), AUCprereF(i,4),AUCprereF(i,5)] = SVMensemble(training,trtarget,testing,testingtar, kernels{i},1);
%    [X1(:,i),Y1(:,i),T,AUCprereF(i,1)] = ROC(scores,testingtar);
    [X1(:,i),Y1(:,i),T,AUCprereF(i,1)] = perfcurve(testingtar,scores,1);
end

enAUCprereF = zeros(4,5);%AUC,accuracy,Precision, Recall,F
enX1 = zeros(length(testingtar)+1,4);
enY1 = zeros(length(testingtar)+1,4);
scores = zeros(length(testingtar)+1,4);
i = 1;
[label, score, enAUCprereF(i,2), enAUCprereF(i,3), enAUCprereF(i,4),enAUCprereF(i,5)] = SVMensemble(training,trtarget,testing,testingtar, {'RBF';'RBF';'RBF'},3);
[enX1(:,i),enY1(:,i),T,enAUCprereF(i,1)] = perfcurve(testingtar,score,1);
i = 2;
[label, score, enAUCprereF(i,2), enAUCprereF(i,3), enAUCprereF(i,4),enAUCprereF(i,5)] = SVMensemble(training,trtarget,testing,testingtar, {'RBF';'polynomial';'linear'},3);
[enX1(:,i),enY1(:,i),T,enAUCprereF(i,1)] = perfcurve(testingtar,score,1);
i = 3;
[label, score, enAUCprereF(i,2), enAUCprereF(i,3), enAUCprereF(i,4),enAUCprereF(i,5)] = SVMensemble(training,trtarget,testing,testingtar, {'polynomial';'linear'},2);
[enX1(:,i),enY1(:,i),T,enAUCprereF(i,1)] = perfcurve(testingtar,score,1);
i = 4;
[label, score, enAUCprereF(i,2), enAUCprereF(i,3), enAUCprereF(i,4),enAUCprereF(i,5)] = SVMensemble(training,trtarget,testing,testingtar, {'RBF';'linear'},2);
[enX1(:,i),enY1(:,i),T,enAUCprereF(i,1)] = perfcurve(testingtar,score,1);


plot(X1(:,1),Y1(:,1),X1(:,3),Y1(:,3),enX1(:,1),enY1(:,1),enX1(:,2),enY1(:,2))
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC curve');
legend(['RBF/RBF/RBF(AUC = ' num2str(AUCprereF(1,1)) ' )'],...
    ['linear(AUC = ' num2str(AUCprereF(3,1)) ' )'],['RBF(AUC = ' num2str(enAUCprereF(1,1)) ' )'],...
    ['RBF/polynomial/linear(AUC = ' num2str(enAUCprereF(2,1)) ' )']);
