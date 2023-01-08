function [ N_update ] = inullspace( D1,D2,ND,NullDegree)

DD = [D1'*ND D2'];

NDD = null(DD);

NDD1 = ND*NDD(1:size(ND,2),1:end);

NDD2 = NDD(size(ND,2)+1:size(NDD,1),1:end);

N_update = [NDD1;NDD2];

end

