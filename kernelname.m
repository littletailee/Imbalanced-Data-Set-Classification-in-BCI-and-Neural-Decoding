function [ kernel ] = kernelname( position )
    if position == 1
        %kernel = 'quadratic';
        kernel = 'rbf';
    end
    if position == 2
        kernel = 'rbf';
    end
    if position == 3
        kernel = 'polynomial';
    end
    if position == 4
        kernel = 'mlp';
    end
end

