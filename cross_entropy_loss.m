function [loss] = cross_entropy_loss(scores, correct_class)

loss = -log(exp(scores(correct_class))/sum(exp(scores)));

end

