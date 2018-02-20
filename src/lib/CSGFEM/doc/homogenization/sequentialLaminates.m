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
% @param[in] Ne         number of evenly spaced angles, alpha, of directions, e,
%                       to try in [0, pi) 
% @return    AStars     (Nt * Ne)^p homogenized elasticity tensors,
%                       with the nth flattened tensor in AStars(:, :, n)
%            params     nth tensor's choices for alpha_i, theta_i with
%                       params(:, n) =
%                                 (alpha_1, ..., alpha_p, theta_1, ... theta_p)'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [AStars, params] = sequentialLaminates(lamA, muA, lamB, muB, p, Nt, Ne)
    A = [lamA+2*muA, lamA,       0;
         lamA,       lamA+2*muA, 0;
         0,          0,          2*muA];
    B = [lamB+2*muB, lamB,       0;
         lamB,       lamB+2*muB, 0;
         0,          0,          2*muB];
    function fA = fAGen(e)
        e1e1 = e(1) * e(1); e2e2 = e(2) * e(2); e1e2 = e(1) * e(2);
        fA = (1 / (4 * muA)) * [4*e1e1, 0,           2*e1e2;
                                0,           4*e2e2, 2*e1e2;
                                2*e1e2,      2*e1e2, e1e1+e2e2] + ...
            (1 / (2 * muA + lamA) - 1 / muA) * [e1e1*e1e1, e1e1*e2e2, e1e1*e1e2;
                                                e1e1*e2e2, e2e2*e2e2, e1e2*e2e2;
                                                e1e1*e1e2, e1e2*e2e2, e1e1*e2e2];
    end

    % Initialize fA(e_i) and theta_i with the first choice
    % (e_i = [1, 0], theta_i = 1/(Nt + 1))
    eSteps = ones(p, 1);
    thetaSteps = ones(p, 1);
    fA = repmat(fAGen([1; 0]), [1, 1, p]);
    theta = (1 / (Nt + 1)) * thetaSteps;

    assert(abs(det(B - A)) > 1e-3, 'singular B - A');
    BmAinv = (B - A)^-1;

    NeP = Ne^p;
    NtP = Nt^p;
    AStars = zeros(3, 3, NeP*NtP);
    params = zeros(2 * p, NeP*NtP);

    for eIt = 1:NeP
        for thetaIt = 1:NtP
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
            assert(abs(det(BmAinv + fAComb)) > 1e-3, 'singular BmAinv + fAComb');
            AStars(:, :, (eIt-1) * NtP + thetaIt) = A + angleProd * (BmAinv + fAComb)^-1;
            params(   :, (eIt-1) * NtP + thetaIt) = [(eSteps-1)*pi/Ne; theta];

            % Increment the collection of p "theta indices", also updating theta
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

        % Increment the collection of p "e indices", also updating fA
        for i = 1:p
            eSteps(i) = eSteps(i) + 1;
            if eSteps(i) > Ne
                eSteps(i) = 1;
                alphai = (eSteps(i) - 1) * pi / Ne;
                fA(:, :, i) = fAGen([cos(alphai); sin(alphai)]);
            else
                alphai = (eSteps(i) - 1) * pi / Ne;
                fA(:, :, i) = fAGen([cos(alphai); sin(alphai)]);
                break;
            end
        end
    end
end
