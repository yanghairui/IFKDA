function [model, time] = KNFSTupdate(K1,K2, labels , model)
% [model, time] = KNFSTupdate(K1,K2, labels , model): Incremental Kernel Null Space Discriminant Analysis for Novelty Detection
% Input:
%       - K1: the kernel matrix of X and Y
%       - K2: the kernel matrix of Y and Y
%       - labels: labels for Y
%       - model: current model for null space DA
%
% Output:
%       - time: computational time for update
%       - model: updated model
%
% Page:
%
%      See: http://www.icst.pku.edu.cn/zlian/IKNLDA/
%
%Reference:
%
%   Juncheng Liu, Zhouhui Lian, Yi Wang, Jianguo Xiao
%   "Incremental Kernel Null Space Discriminant Analysis for Novelty
%   Detection", IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2017.
%   
%
%   For the original null space DA see:
%   Paul Bodesheim and Alexander Freytag and Erik Rodner and Michael Kemmler and Joachim Denzler. 
%   Kernel Null Space Methods for Novelty Detection. IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2013.
%
%
%   Written by Juncheng Liu (liujuncheng@pku.edu.cn)


tic
    classes = unique(labels);
%     model.classes = classes;
    model.labels = [model.labels labels];
    model.classes = unique(model.labels);
    model.nclass  = model.nclass + length(unique(labels));
    
    N = model.N;
    l = size(K1,2);

time(1) = toc ;
tic

    L = zeros(l,l);
    for i=1:length(classes)

       L(labels==classes(i),labels==classes(i)) = 1/sum(labels==classes(i));

    end

    D1 = (model.Lambda)'*K1*(eye(l)-L);
    
time(2) =  toc ;
tic
    
    rho1 = sqrt(l/(N*(N+l)));
    rho2 = -sqrt(N/(l*(N+l)));
    
    Xi1 = [zeros(N,l) rho1.*ones(N,1)];
    Xi2 = [[eye(l) - (1/l).*ones(l,l)] rho2.*ones(l,1)];

    SK1 = centerKernelMatrix(K1);
    Gamma = model.Lambda'*[SK1 rho1*sum(model.K,2)+rho2*sum(K1,2)];
    
    Omega1 = Xi1 - model.Lambda*Gamma;
    Omega2 = Xi2;
    
time(3) = toc;
tic
    
    [Q, Delta] = eig(Omega1'*model.K*Omega1 + Omega2'*K1'*Omega1 + Omega1'*K1*Omega2 + Omega2'*K2*Omega2);
    
    Q = real(Q);
    Delta = real(Delta);
    
    basisvecsValues = diag(Delta);
    
    basisvecs = Q(:,basisvecsValues >= 1e-12);
    basisvecsValues = basisvecsValues(basisvecsValues >= 1e-12);
    basisvecsValues = diag(1./sqrt(basisvecsValues));
    
    Theta = [Omega1;Omega2]*basisvecs*basisvecsValues;
    
    D2 = Theta'*[K1;K2]*(eye(l)-L);
    
time(4) = toc;
tic
       [ model.U ] = inullspace( D1 ,D2 , model.U, model.NullDegree);
       model.NullDegree = model.NullDegree + length(unique(labels));


     
time(5) = toc;
tic  
     
    model.Lambda = [[model.Lambda; zeros(size(Theta,1) - size(model.Lambda,1), ...
                    size(model.Lambda,2)) ]  Theta];
    
time(6) = toc;    

tic
    model.proj = model.Lambda*model.U;
    
    model.N = model.N + l;
    model.K = [model.K K1; K1' K2];
    model.target_points = zeros( model.nclass ,size(model.proj,2) );

    for c=1:model.nclass

      id = model.labels == model.classes(c);
      model.target_points(c,:) = mean(model.K(id,:)*model.proj); 
       
    end
time(7) = toc; 

    
end 


%*************************
function centeredKernelMatrix = centerKernelMatrix(kernelMatrix)
  n = size(kernelMatrix, 2);

  columnMeans = mean(kernelMatrix,2); 

  centeredKernelMatrix = kernelMatrix;

  for k=1:n

    centeredKernelMatrix(:,k) = centeredKernelMatrix(:,k) - columnMeans;

  end

end 



