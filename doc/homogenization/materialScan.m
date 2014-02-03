%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Scan over valid material pairs (A, B) and sample their space of rank p
% sequential laminates (p = 1, 2, 3). Write the laminates' parameters and
% homogenized elasticity tensors to 'h$vA_$EB_$vB_p.mat' for visualization.
% We assume WLOG that A's Young modulus, EA = 1. We evenly space EB in (1, 10)
% and choose both A and B's Poisson ratios evenly spaced in (-1, .5). See
% 'homogenization.pdf' for a explanation of how the material space is reduced to
% this space.
% @param[in]    Ne      number of Young's modulus samples (for EB)
% @param[in]    Nv      number of Poisson ratio samples (for vA and vB)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function materialScan(Ne, Nv)
    Erange = linspace(1, 10, Ne + 1);  % Ne evenly spaced in (1, 10]
    vrange = linspace(-1, .5, Nv + 2); % Nv evenly spaced in (-1, .5)
    Erange = Erange(2:end);
    vrange = vrange(2:end-1);

    lambda = @(E, v) (E * v)/((1 + v) * (1 - v));
    mu     = @(E, v) E / (2 * (1 + v));
    kappa  = @(E, v) E / (2 * (1 - v));

    EA = 1;
    for vA = vrange
        lamA = lambda(EA, vA);
        muA  = mu(EA, vA);
        kA   = kappa(EA, vA);
        for EB = Erange
            for vB = vrange
                lamB = lambda(EB, vB);
                muB  = mu(EB, vB);
                kB   = kappa(EB, vB);
                [vA EB vB]
                % Check material pair. We can't have A == B because EB > EA.
                if ((kA > kB) || (muA > muB))
                    continue;
                end
                for p = 1:3
                    [AStars, params] = ...
                        sequentialLaminates(lamA, muA, lamB, muB, p, 10, 10);
                    outName = sprintf('h%f_%f_%f_%i.mat', vA, EB, vB, p);
                    save(outName, 'AStars', 'params', 'EA', 'vA', 'EB', 'vB'); 
                end
            end
        end
    end
end
