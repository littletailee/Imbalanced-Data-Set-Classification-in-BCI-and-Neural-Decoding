%G为关于x,y的函数,求最大值
minx = 0.1;
maxx = 2;
miny = 0;
maxy = 2;
u1 = 100;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%迭代次数
dx = maxx - minx;
dy = maxy - miny;
x = zeros(1,4);
y = zeros(1,4);
x(1) = minx;
x(2) = maxx;
y(1) = miny;
y(2) = maxy;
for k = 1:u1
    k
    x(3) = (2*x(1) + x(2))/3;
    x(4) = (x(1) + 2*x(2))/3;
    y(3) = (2*y(1) + y(2))/3;
    y(4) = (y(1) + 2*y(2))/3;
    x(1)
    y(1)
    maxs1 = -Gx(x(1),y(1));%%%%%%%%%%%%%%%这里需要改
    maxgx = x(1);
    maxgy = y(1);
    for r = 1:4
        for s = 1:4
            w = -Gx(x(r),y(s));%%%%%%%%%%%这里需要改
            if w > maxs1
                maxs1 = w;
                maxgx = x(r);
                maxgy = y(s);
            end
        end
    end
    dx = dx*2/3;
    dy = dy*2/3;
    if maxgx - dx/2 < minx
        x(1) = minx;
        x(2) = minx + dx;
    else
        if maxgx + dx/2 > maxx
            x(2) = maxx;
            x(1) = maxx - dx;
        else
            x(1) = maxgx - dx/2;
            x(2) = maxgx + dx/2;
        end
    end
    if maxgy - dy/2 < miny
        y(1) = miny;
        y(2) = miny + dy;
    else
        if maxgy + dy/2 > maxy
            y(2) = maxy;
            y(1) = maxy - dy;
        else
            y(1) = maxgy - dy/2;
            y(2) = maxgy + dy/2;
        end
    end
end
maxgx%%%%%%%%%%%%%%输出的结果
maxgy%%%%%%%%%%%%%%输出的结果