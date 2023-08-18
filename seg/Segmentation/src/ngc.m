function cidxs = ngc(X)

% This is a simple wrapper to perform binary clustering using the
% Normalized Graph Cuts (NGC) method, which was originally published on the
% following reference:
% Jiambo Shi and Jitendra Mali, "Normalized Cuts and Image Segmentation",
% IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
% 2000.

% This wrapper makes use of some functions downloaded from the Jiambo Shi'
% web page (https://www.cis.upenn.edu/~jshi/software/).

% INPUT:
% X     - KxN matrix in which each column corresponds to a K-dimensional data point. 

% OUTPUT:
% cidxs - N-dimensional vector in which each element provides the cluster index assigned from the correspondent data point.

[W, ~] = compute_relation(X);
[y, ~, ~] = ncutW(W, 2); 
cidxs = 2*y(:,1) + 1*y(:,2);

end
