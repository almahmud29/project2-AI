function dy = d_softmax(v, temp)

for i = 1:1:10
    if i==temp
        j = temp;
        dy(i) = (-exp(v(i))*exp(v(j)))/((sum(exp(v)).^2));
        % dy(i) = (exp(v(i))*(sum(exp(v))-exp(v(i))))/((sum(exp(v)).^2));
    else
        % j = temp;
        % dy(i) = (-exp(v(i))*exp(v(j)))/((sum(exp(v)).^2));
        dy(i) = (exp(v(i))*(sum(exp(v))-exp(v(i))))/((sum(exp(v)).^2));
    end
end


end