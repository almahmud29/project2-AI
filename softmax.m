function y = softmax(v)

for i=1:1:10
    y(i) = (exp(v(i)))/(sum(exp(v)));
end

end