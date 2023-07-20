clear
close all
load MnistConv.mat


% define useful parameters %

num_tr = 1000; % sample of training data
num_te = 9000; % sample of testing data

LR = 0.005;
% biases = zeros(1,10);

training_data = Images(:,:,1:1000); % training data
training_label = Labels(1:1000, 1); % training_label
test_data = Images(:,:,1001:10000); % test data
test_label = Labels(1001:10000, 1); % test_label

w_con = randn(3, 3, 8)/9;
w_original_con = w_con;
w_softmax = randn(10, 1353)/1353;
w_original = w_softmax;

for img = 1:1:length(training_data)

    Label = training_label(img);
    x_image = training_data(:,:,img) - 0.5;
    for i=1:1:8
        filter = w_con(:,:,i);
        filter = rot90(squeeze(filter), 2);
        y1(:,:,i) = conv2(x_image, filter, 'valid');
%         y1(:,:,i) = conv2(training_data(:,:,img), filter, 'valid');
    end

    % Step 2: Max Pooling

    xx = [];
    Max_Pool = zeros(26, 26, 8);
    for i=1:1:8
        for ii=1:1:13
            for jj=1:1:13
                max_value = max(max(y1((ii-1)*2+1:ii*2,(jj-1)*2+1:jj*2,i)));
                y2(ii,jj,i) = max_value;

                [rr cc] = find((y1((ii-1)*2+1:ii*2,(jj-1)*2+1:jj*2,i))==max_value);

                xx = [xx; rr(1) cc(1) ii jj i max_value];       % needed
                    % update max pooling for backward calculation
                Max_Pool((ii-1)*2+rr(1),(jj-1)*2+cc(1),i) = max_value;

            end
        end
    end
    
    % Step 3: Fully connection
    
    y3 = y2(:);         % convert to a column vector
    y3 = [y3; 1];       % Add bias
    v = y3'*w_softmax';

    %     v = y3'*w_softmax' + biases;
    % Step 4: Softmax Function
    
    y = softmax(v);     % apply softmax function
    
    L = -log(y);         % cross entropy loss

    % Step 5: Final Output
    
    [output, label] = max(y);
    
    
    % ======== Backward Propagation ======== %
    
    % error calculation

    % error calculation

    temp = Label;

    for i = 1:1:10
        if i==temp
            error(i) = -1/y(temp);
        else
            error(i) = 0;
        end
    end

    dy = d_softmax(v, temp); % derivative of softmax

    dw1 = error.*dy.*y3; % 11 June
    w_softmax = w_softmax - dw1';
    
%     w_softmax = w_softmax + dw1'; % Previous
    % Step 4: Update max-pooling
    
    dw = sum((error.*dy.*w_softmax')');
    y3_back = dw';
    
    y2_1d = y3_back(1:1352);

    for i = 1:1:length(y2_1d)
        if xx(i,6) ~= 0
            xx(i,6) = y2_1d(i);
        end
    end
    
    M_Pool = zeros(26, 26, 8);
    Z = 1;
    for i = 1:1:8
        for ii = 1:1:13
            for jj = 1:1:13
                M_Pool((ii-1)*2+rr(1),(jj-1)*2+cc(1),i) = xx(Z,6); % Updating Max_Pooling function
                Z = Z+1;
            end
        end
    
    end

    d_out_d_f = zeros(3, 3, 8);
    
    image = training_data(:,:,img);
    for k=1:1:8
        for ii = 1:1:26
            for jj = 1:1:26
                d_out_d_f(:,:,k) = d_out_d_f(:,:,k) + image(ii:ii+2, jj:jj+2)*M_Pool(ii,jj,k); % change M_pool to Max_pool
            end
        end
    
    end
    
    w_con = w_con - d_out_d_f*LR;


    mse(img) = sum(mean(error'.^2));

end


%% results from original untrained weight

for im=1:num_tr
    x = training_data;
    d = training_label;
    for i=1:1:8
        filter = w_original_con(:,:,i);
        filter = rot90(squeeze(filter), 2);
        y1(:,:,i) = conv2(training_data(:,:,im), filter, 'valid');
    end

    for i=1:1:8
        for ii=1:1:13
            for jj=1:1:13
                max_value = max(max(y1((ii-1)*2+1:ii*2,(jj-1)*2+1:jj*2,i)));
                y2(ii,jj,i) = max_value;
            end
        end
    end

    y3 = y2(:);         % convert to a column vector
    y3 = [y3; 1];       % add bias % 11 June

    v = y3'*w_original';

    y = softmax(v);     % apply softmax function

    L = -log(y);         % cross entropy loss

    % Step 5: Final Output

    [output, label] = max(y);
    out_original(im) = label;
end

o = out_original';


%% results from well trained weight

for im=1:num_tr
    x = training_data;
    d = training_label;
    for i=1:1:8
        filter = w_con(:,:,i);
        filter = rot90(squeeze(filter), 2);
        y1(:,:,i) = conv2(training_data(:,:,im), filter, 'valid');
    end

    for i=1:1:8
        for ii=1:1:13
            for jj=1:1:13
                max_value = max(max(y1((ii-1)*2+1:ii*2,(jj-1)*2+1:jj*2,i)));
                y2(ii,jj,i) = max_value;
            end
        end
    end

    y3 = y2(:);         % convert to a column vector
    y3 = [y3; 1];       % add bias % 11 June
    


    v = y3'*w_softmax';

    y = softmax(v);     % apply softmax function

    L = -log(y);         % cross entropy loss

    % Step 5: Final Output

    [output, label] = max(y);
    out_original(im) = label;
end

o_well_trained = out_original';


%% Plotting

figure
for ii = 1:num_tr
    x = ii;
    y = mse(ii)';
%     plot(x, y);
    plot(x, y, 'r*', 'LineWidth', 3, 'MarkerSize', 4);
    hold on;
end


%% results from original untrained weight (test)

for im=1:num_te
    x = test_data;
    d = test_label;
    for i=1:1:8
        filter = w_original_con(:,:,i);
        filter = rot90(squeeze(filter), 2);
        y1(:,:,i) = conv2(test_data(:,:,im), filter, 'valid');
    end

    for i=1:1:8
        for ii=1:1:13
            for jj=1:1:13
                max_value = max(max(y1((ii-1)*2+1:ii*2,(jj-1)*2+1:jj*2,i)));
                y2(ii,jj,i) = max_value;
            end
        end
    end

    y3 = y2(:);         % convert to a column vector
    y3 = [y3; 1];       % add bias % 11 June

    v = y3'*w_original';

    y = softmax(v);     % apply softmax function

    L = -log(y);         % cross entropy loss

    % Step 5: Final Output

    [output, label] = max(y);
    out_original(im) = label;
end

o_test = out_original';


%% results from well trained weight (test)

for im=1:num_te
    x = test_data;
    d = test_label;
    for i=1:1:8
        filter = w_con(:,:,i);
        filter = rot90(squeeze(filter), 2);
        y1(:,:,i) = conv2(test_data(:,:,im), filter, 'valid');
    end

    for i=1:1:8
        for ii=1:1:13
            for jj=1:1:13
                max_value = max(max(y1((ii-1)*2+1:ii*2,(jj-1)*2+1:jj*2,i)));
                y2(ii,jj,i) = max_value;
            end
        end
    end

    y3 = y2(:);         % convert to a column vector
    y3 = [y3; 1];       % add bias % 11 June
    


    v = y3'*w_softmax';

    y = softmax(v);     % apply softmax function

    L = -log(y);         % cross entropy loss

    % Step 5: Final Output

    [output, label] = max(y);
    out_original(im) = label;
end

o_test_well_trained = out_original';

%% Plot Test

figure
for ii = 1:num_te
    x = ii;
    y = o_test_well_trained(ii);
%     plot(x, y);
    plot(x, y, 'r*', 'LineWidth', 3, 'MarkerSize', 4);
    hold on;
end

%% 
figure
for ii = 1:num_te
    x = ii;
    y = o_test(ii);
%     plot(x, y);
    plot(x, y, 'r*', 'LineWidth', 3, 'MarkerSize', 4);
    hold on;
end
%%
figure
plot(1:length(o_test),o_test,1:length(o_test_well_trained),o_test_well_trained)

%%
figure
plot(1:length(o),o,1:length(o_well_trained),o_well_trained)

%%
for i = 1:1:1000;
    Accuracy_o_train(i) = isequal( o(i),training_label(i) );
end
 temp = sum(Accuracy_o_train);

 Accuracy_o_train = (temp/1000)*100;

%%
for i = 1:1:1000;
    Accuracy_o_w_train(i) = isequal( o_well_trained(i),training_label(i) );
end
 temp = sum(Accuracy_o_w_train);

 Accuracy_o_w_train = (temp/1000)*100;

%%
for i = 1:1:9000;
    Accuracy_o_test(i) = isequal( o_test(i),test_label(i) );
end
temp = sum(Accuracy_o_test);

Accuracy_o_test = (temp/9000)*100;

%%
for i = 1:1:9000;
    Accuracy_o_test_wt(i) = isequal( o_test_well_trained(i),test_label(i) );
end
temp = sum(Accuracy_o_test_wt);

Accuracy_o_test_wt = (temp/9000)*100;
