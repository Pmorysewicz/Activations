D=4;%4 inputs
M=3;%3 hidden units, relu sigmoid and tanh?
K=2;%2 outputs

load weights_samples.mat
input = [10,1,2,3];
weight1 = [0.5, 0.6, 0.4, 0.3; 0.02, 0.25, 0.4, 0.3; 0.82, 0.1, 0.35, 0.3];
weight2 = [0.7, 0.45, 0.5; 0.17, 0.9, 0.8 ];
%weight them 
w1 = input .* weight1(1,:);
w2 = input .* weight1(2,:);
w3 = input .* weight1(3,:);
%same thing but in one chunk idk whats easier to use yet
weighted1 = input .* weight1;%spits 12 numbers weighted input 
%do tanh activation separate them by rows so i can sum them 
bias = 0;%change to 0 cus no bias lol
a1 = tanh(sum(w1)+ bias);
z2 = tanh(sum(w2)+ bias)%this is "z2"
a3 = tanh(sum(w3)+ bias);
%z2 = [a1,a2,a3];%part 1 Q2 ans i think [1 , 0.9963, 1]

%------------part1 q 3 ------------------
%do relu activation
b1 = max(0,sum(w1)+ bias);
b2 = max(0,sum(w2)+ bias);
b3 = max(0,sum(w3)+ bias);
brelu = [b1,b2,b3];
weightedrelu1 = brelu .* weight2(1,:);
weightedrelu2 = brelu .* weight2(2,:);
%sigmoid activation
ws1 = sum(weightedrelu1);
ws2 = sum(weightedrelu2);
yt1 = sum(1./(1 + exp(-ws1)));%3.7977
yt2 = sum(1./(1 + exp(-ws2)));%3.7483
y = [yt1,yt2] 

