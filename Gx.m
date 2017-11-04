function [g,avg] = Gx(xg,yg)
load('BFS.mat');
Datatrain = Data;
load('BFS_online.mat');
Datatest = data_online;
Cmr = zeros(4,720);
grno = zeros(1,4);
ervalue = zeros(1,3);
trainingset = [];
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
        Datatrain{k,1}(r,1:30) = guiyi(Datatrain{k,1}(r,1:30),xg,yg);%%%%%%%%%%%%%%修改处
        Datatrain{k,1}(r,31:60) = guiyi(Datatrain{k,1}(r,31:60),xg,yg);%%%%%%%%%%%%%%修改处
    end
    for r = 1:600
        Datatrain{k,2}(r,1:30) = guiyi(Datatrain{k,2}(r,1:30),xg,yg);%%%%%%%%%%%%%%修改处
        Datatrain{k,2}(r,31:60) = guiyi(Datatrain{k,2}(r,31:60),xg,yg);%%%%%%%%%%%%%%修改处
    end
end
for k = 4
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
                    Datatest1((u3 - 1)*12 + u1,1:30) = guiyi(Datatest1((u3 - 1)*12 + u1,1:30),xg,yg);%%%%%%%%%%%%%%修改处
                    Datatest1((u3 - 1)*12 + u1,31:60) = guiyi(Datatest1((u3 - 1)*12 + u1,31:60),xg,yg);%%%%%%%%%%%%修改处
                end
            end
        end
        Datatest2((r - 1)*144 + 1:r*144,:) = Datatest1(:,:);
    end
    [Cm,value,groupno,avg2,tresult2,position2] = unbalancetest(Datatrain,Datatest2,k,1,16);
    Cmr(k - 1,:) = Cm(1,:);
    tresult3(k - 1,:) = tresult2(:);
    position(k - 1,:) = position2(:);
    grno(k - 1) = groupno;
    ervalue(k - 1) = value;
    avg = avg2(3,:);
end
tresult3
grno
u = zeros(2,16);
u(1,1:16) = avg;
u(2,1:16) = position(3,:);
g = min(u(1,:));