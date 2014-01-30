%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sequentialLaminates.m
% 01/30/2014 - Julian Panetta 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Explore the space of homogenized sequential laminates, with isotropic material
% A laminating isotropic material B p times in a discretized set of proportions
% and directions.
% @param[in] lamA, muA  Lame parameters for material A
% @param[in] lamB, muB  Lame parameters for material B
% @param[in] p          number of lamination steps
% @param[in] Nt         number of evenly spaced proportions, theta, to try in
%                       (0, 1)
% @param[in] Ne         number of evenly spaced angles of directions, e, to try
%                       in [0, 2pi) 
% @return    AStars     (Nt * Ne)^p homogenized elasticity tensors,
%                       with the ith flattened tensor in AStars(:, :, i)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function AStars = sequentialLaminates(lamA, muA, lamB, muB, p, Nt, Ne)
    A = [lamA+2*muA, lamA,       0;
         lamA,       lamA+2*muA, 0;
         0,          0,          2*muA]
    B = [lamB+2*muB, lamB,       0;
         lamB,       lamB+2*muB, 0;
         0,          0,          2*muB]

    fA1 = @(e) [4*e(1)*e(1), 0,           2*e(1)*e(2);
                0,           4*e(2)*e(2), 2*e(1)*e(2);
                2*e(1)*e(2), 2*e(1)*e(2), e(1)*e(1)+e(2)*e(2)];

    fA2 = @(e) [e(1)*e(1)*e(1)*e(1), e(1)*e(1)*e(2)*e(2), e(1)*e(1)*e(1)*e(2);
                e(1)*e(1)*e(2)*e(2), e(2)*e(2)*e(2)*e(2), e(1)*e(2)*e(2)*e(2);
                e(1)*e(1)*e(1)*e(2), e(1)*e(2)*e(2)*e(2), e(1)*e(1)*e(2)*e(2)];

    fAGen = @(e) fA1(e) / (4 * muA) + (1 / (2 * muA + lamA) - 1 / muA) * fA2(e);

    eSteps = ones(p, 1);
    thetaSteps = ones(p, 1);
    fA = zeros(3, 3, p);
    theta = zeros(p, 1);

    BmAinv = (B - A)^-1;

    AStars = zeros(3, 3, Ne^p*Nt^p);

    for eIt = 1:(Ne^p)
        for thetaIt = 1:(Nt^p)
            % Evaluate the linear combination of fA(e_i)s
            fAComb = zeros(3);
            for i = 1:p
                fACoeff = theta(i);
                for j = 1:i-1
                    fACoeff = fACoeff * (1 - theta(j));
                end
                fAComb = fAComb + fACoeff * fA(:, :, i);
            end

            % Solve for A^*_p
            angleProd = 1;
            for i = 1:p
                angleProd = angleProd * (1 - theta(i));
            end
            AStars(:, :, eIt * thetaIt) = A + angleProd * (BmAinv + fAComb)^-1;

            % Increment the collection of p "theta indices"
            for i = 1:p
                thetaSteps(i) = thetaSteps(i) + 1;
                if thetaSteps(i) > Nt
                    thetaSteps(i) = 1;
                    theta(i) = thetaSteps(i) / (Nt + 1);
                else
                    theta(i) = thetaSteps(i) / (Nt + 1);
                    break;
                end
            end
        end

        % Increment the collection of p "e indices"
        for i = 1:p
            eSteps(i) = eSteps(i) + 1;
            if eSteps(i) > Ne
                eSteps(i) = 1;
                alphai = (eSteps(i) - 1) * 2 * pi / Ne;
                fA(:, :, i) = fAGen([cos(alphai); sin(alphai)]);
            else
                alphai = (eSteps(i) - 1) * 2 * pi / Ne;
                fA(:, :, i) = fAGen([cos(alphai); sin(alphai)]);
                break;
            end
        end
    end
end
