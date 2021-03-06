function [loss] = hinge_loss(scores, correct_class)


%     scores is a 4x1 set of predicted scores, one score for each class, for some sample, and
%     correct_class is the correct class for that same sample.

%hinge loss formula loss = SUM max(0, Sj(scores) - Syi(correct class), +1)
loss = 0;
for i = 1:size(scores)
loss = loss + (max(0, scores(i) - scores(correct_class) + 1));
end
loss = loss / length(scores);
%loss = loss / length(scores);
end

