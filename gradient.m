
%load weights_samples.mat

%cur loss, new loss plus updated loss

Current = W1;% store it for later 
Deriv = [];
h=0.0001;
   
EDelta = [];
WV = W1(:);


for i = 1:length(WV)%for each weight 
    WV = Current;
    
    %calculate scores every iteration?
    s1 = WV*x1;
    s2 = WV*x2;
    s3 = WV*x3;
    s4 = WV*x4;
    
   %calulate current los
   Current_loss1 = hinge_loss(s1, 1);
   Current_loss2 = hinge_loss(s2, 2);
   Current_loss3 = hinge_loss(s3, 3);
   Current_loss4 = hinge_loss(s4, 4);
   %average them 
   Current_loss = (Current_loss1 + Current_loss2 + Current_loss3 + Current_loss4)/4;
   %size(Current_loss)
   %perturb = from google Perturbation means adding noise, usually to the training data but sometimes to the learnt parameters.
   %is it just add h then?
   %fix for now mjust to be sure it doesnt break here
   Current_reshaped = Current;
   Current_reshaped(i) = Current_reshaped(i) + h;
   %Use reshape to reshape any intermediate W1_plus_h back into a 4x25 matrix when you need to compute the scores s = W*x.
   
   reshape(Current_reshaped, 4,25);
   
    % new scores 
    ns1 = Current_reshaped*x1;
    ns2 = Current_reshaped*x2;
    ns3 = Current_reshaped*x3;
    ns4 = Current_reshaped*x4;
    
   %compute new loss
   New_loss1 = hinge_loss(ns1, 1);
   New_loss2 = hinge_loss(ns2, 2);
   New_loss3 = hinge_loss(ns3, 3);
   New_loss4 = hinge_loss(ns4, 4);
   %average them 
   New_loss = (New_loss1 + New_loss2 + New_loss3 + New_loss4)/4;
   %    size(New_loss)
   %    size(Current_loss)
   % compute change in E (formula)
   % (New_loss - Current_loss)/ h
   Derivative = (New_loss - Current_loss)/h;
   %ChangeW = ( Current_reshaped / W1 ); % current holds orginal newW/W 
   %find gradient with respect to W 
   % EDelta = sigma E/W(i)? 
   EDelta = [EDelta, Derivative];%vector of EDelta
   % is it just just update W? 
  % Current(i) = Current(i) + h;% do i need it?
    
end
%size(W1)
%size(WV)
%size(Current)
%size(EDelta)
%size(WV(:))
%size(EDelta')%need to transpose it 
% since we iterate we will have the loss from the last iteration?
    %WV  = EDelta / (WV - W1);
    learning =  0.001;
    
    W1 = WV(:)- (learning* EDelta') ;% use WV as vector
    %size(WV)
    W1 = reshape(W1, [4,25]);
    %size(WV)
    s1 = W1*x1;
    s2 = W1*x2;
    s3 = W1*x3;
    s4 = W1*x4;
    
   %calulate final loss
   Final_loss1 = hinge_loss(s1, 1);
   Final_loss2 = hinge_loss(s2, 2);
   Final_loss3 = hinge_loss(s3, 3);
   Final_loss4 = hinge_loss(s4, 4);
   Final_loss = (Final_loss1 + Final_loss2 + Final_loss3 + Final_loss4)/4

%lerning 

% find updated loss


