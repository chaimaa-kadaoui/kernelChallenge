function [ maxList ] = anms( img, nmax,c )
%Implementing adaptive non maxima suppression
%   img contains the corenerness measure of the image
%   nmax is the number of maxima we want
%   c is the algorithm parameter (typical value : 0.9)

INFTY = 10^10;

[n,m] = size(img);
[V,ind] = sort(img(:),1,'descend');
[I, J] = ind2sub([n,m], ind);

processedPoints = zeros(nmax,3);
processedPoints(1,:) = [I(1) J(1) INFTY];
detectedPoints = V;

for i = 2:length(detectedPoints)
  dist_current_to_processedPoints = (I(i)-processedPoints(1:i-1,1)).^2 + (J(i)-processedPoints(1:i-1,2)).^2;
  ind_with_criterion = V(i) < c*V(1:i-1); 
  min_dist = INFTY;
  if nnz(ind_with_criterion) ~= 0
    min_dist = min(dist_current_to_processedPoints(ind_with_criterion));
  end
  processedPoints(i,:) = [I(i),J(i),min_dist];
end

[Y,final_indice]=sort(processedPoints(:,3),1,'descend');
maxList = processedPoints(final_indice,:);
maxList = maxList(1:nmax,1:2);
end

