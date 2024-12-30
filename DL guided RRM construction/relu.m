function x=relu(x)
for i=1:size(x,1)
    for j=1:size(x,2)
        if x(i,j)<0
            x(i,j)=0;
        end
    end
end
