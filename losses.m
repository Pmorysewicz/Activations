load weights_samples.mat
%edit it has to be W*x lmao 
s1 = W1*x1;
s2 = W2 *x1;
s3 = W3*x1;

s4 = W1* x2;
s5 = W2 *x2;
s6 = W3 *x2;

s7 = W1 *x3;
s8 = W2 *x3;
s9 = W3 * x3;

s10 = W1 *x4;
s11 = W2 *x4;
s12 = W3 *x4;

%correct num for x1 is 1 and so on for other ones

hinge_loss1 = hinge_loss(s1,1);
hinge_loss2 = hinge_loss(s2,1);
hinge_loss3 = hinge_loss(s3,1);
hinge_loss4 = hinge_loss(s4,2);
hinge_loss5 = hinge_loss(s5,2);
hinge_loss6 = hinge_loss(s6,2);
hinge_loss7 = hinge_loss(s7,3);
hinge_loss8 = hinge_loss(s8,3);
hinge_loss9 = hinge_loss(s9,3);
hinge_loss10 = hinge_loss(s10,4);
hinge_loss11 = hinge_loss(s11,4);
hinge_loss12 = hinge_loss(s12,4);


cross_loss1 = cross_entropy_loss(s1,1);

cross_loss2 = cross_entropy_loss(s2,1);
cross_loss3 = cross_entropy_loss(s3,1);
cross_loss4 = cross_entropy_loss(s4,2);
cross_loss5 = cross_entropy_loss(s5,2);
cross_loss6 = cross_entropy_loss(s6,2);
cross_loss7 = cross_entropy_loss(s7,3);
cross_loss8 = cross_entropy_loss(s8,3);
cross_loss9 = cross_entropy_loss(s9,3);
cross_loss10 = cross_entropy_loss(s10,4);
cross_loss11 = cross_entropy_loss(s11,4);
cross_loss12 = cross_entropy_loss(s12,4);

w1crossmean = (cross_loss1 + cross_loss4 + cross_loss7 + cross_loss10)/4
w2crossmean = (cross_loss2 + cross_loss5 + cross_loss8 + cross_loss11)/4
w3crossmean = (cross_loss3 + cross_loss6 + cross_loss9 + cross_loss12)/4

w1hingemean = (hinge_loss1 + hinge_loss4 + hinge_loss7 + hinge_loss10)/4
w2hingemean = (hinge_loss2 + hinge_loss5 + hinge_loss8 + hinge_loss11)/4
w3hingemean = (hinge_loss3 + hinge_loss6 + hinge_loss9 + hinge_loss12)/4

