function [D, X] = uitdl(Y, D0, X0)

% This is a modified version of the unsupervised MMI dictionary atom
% selection method. The original implementation was provided by Sam Q. Qiu
% (qiu@cs.umd.edu) and it is based on the following reference:
% Qiang Qiu, Zhuolin Jiang, and Rama Chellappa, "Sparse dictionary-based
% reprsentation and recognition of human actions attributes", International
% Conference on Computer Vision (ICCV) 2011.

% The changes over the original code were made by Eliezer S. Flores
% (eliezersflores@gmail.com) and result in an unsupervised adaptation of
% the Information-Theoretic Dictionary Learning (ITDL) method, which is
% described on the following reference:
% Qiang Qiu, Vishal Patel, and Rama Chellappa, "Information-theoretic
% dictionary learning for image classification", IEEE Transactions on
% Pattern Analysis and Machine Intelligence (TPAMI) 2014.

% INPUTS:
% Y - input signals used to obtain D and X.
% D0 - initial dictionary.
% X0 - initial representation. 

% OUTPUT:
% D: optimized dictionary.

K = size(D0, 2); % number of dictionary atoms
NS = (1:K)'; % non-selected atoms
S = []; % selected atoms
maxMI_set = []; % maximum mutual information set

kappa = cov(full(X0'));
kappa(abs(kappa) < mean(mean(abs(kappa)))) = 0; % truncation for fast computation

% Estimating lambda, the parameter used to balance the compactness and
% representation terms:
max_comp = -inf;
max_repr = -inf;
for k=1:K
    comp = compactness_gain(kappa, k, S, NS);
    if comp > max_comp
        max_comp = comp;
    end
    repr = representation_gain(Y, D0, X0, S, k);
    if repr > max_repr
        max_repr = repr;
    end
end

lambda = max_comp/(max_repr+eps);

% Iteratively evaluating the next best atom: 
for iter = 1:K
    
    maxMI = -inf;
    
    for k = 1:K

        if ismember(k, S)
            continue;
        end
        
        MI = compactness_gain(kappa, k, S, NS) + lambda*representation_gain(Y, D0, X0, S, k);
                        
        if MI > maxMI
            maxMI = MI;
            s = k;
        end
        
    end
    
    maxMI_set = [maxMI_set; maxMI];
    S = [S; s];
    NS(NS == s) = [];
    
end

D = [];
for k = 1:K
    if maxMI_set(k) > 0
        D = [D D0(:,S(k))];
    else
        break;
    end
end

% ensuring more than one atom is selected:
if size(D,2) < 2    
    D = D0(:,S(1:2));
end

X = pinv(D)*Y;
X(X < 0) = 0;

for k = 1:size(X,1)
    val = norm(X(k,:));
    X(k,:) = X(k,:)/val;
    D(:,k) = D(:,k)*val;
end

end

function MI = compactness_gain(kappa, k, S, NS)

if isempty(S)
    sigma1 = kappa(k,k);
    H1 = .5*log2(2*pi*exp(1)*sigma1);
else
    idx1 = find(kappa(k,S)~=0);
    sigma1 = kappa(k,k) - kappa(k,S(idx1))*pinv(kappa(S(idx1),S(idx1)))*kappa(S(idx1),k);
    if sigma1 < 1e-6
        H1 = 0;
    else
        H1 = .5*log2(2*pi*exp(1)*sigma1);
    end
end

if isempty(NS)
    sigma2 = kappa(k,k);
    H2 = .5*log2(2*pi*exp(1)*sigma2);
else
    NS(NS==k) = [];

    idx2 = find(kappa(k,NS)~=0);
    sigma2 = kappa(k,k) - kappa(k,NS(idx2))*pinv(kappa(NS(idx2),NS(idx2)))*kappa(NS(idx2),k);
    if sigma2 < 1e-6
        H2 = 0;
    else
        H2 = .5*log2(2*pi*exp(1)*sigma2);
    end
end

MI = H1 - H2;

end

function MI = representation_gain(Y, D0, X0, S, k)

    MI = representation(Y, D0, X0, [S; k]) - representation(Y, D0, X0, S);

end

function MI = representation(Y, D0, X0, S)

    MI = 0;
    
    if ~isempty(S)
        D = D0(:,S);
        X = X0(S,:);
        for i = 1:size(Y,2)
            r = Y(:,i) - D*X(:,i);
            sigma2 = var(r);
            if sigma2 > 0 
                prob = exp((-1/(2*sigma2))*sum(r.^2));
                if prob > 0
                    MI = MI - prob*log2(prob);
                end
            end
        end
    end
    
end




