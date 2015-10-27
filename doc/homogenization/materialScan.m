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
% @param[in]    vrange  override the evenly-spaced Poisson ratio with a given
%                       array.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function materialScan(Ne, Nv, vrange)
    % Ne evenly spaced Young's moduli in (1, 10]
    Erange = linspace(1, 10, Ne + 1);
    Erange = Erange(2:end);

    if (~exist('vrange', 'var'))
        % Nv evenly spaced Poisson ratios in (-1, .5)
        vrange = linspace(-1, .5, Nv + 2);
        vrange = vrange(2:end-1);
    end

    lambda = @(E, v) (E * v)/((1 + v) * (1 - v));
    mu     = @(E, v) E / (2 * (1 + v));
    kappa  = @(E, v) E / (2 * (1 - v));

    EA = 1;
    matlabpool;
    parfor EBi = 1:size(Erange, 2)
        EB = Erange(EBi);
        for vA = vrange
            lamA = lambda(EA, vA);
            muA  = mu(EA, vA);
            kA   = kappa(EA, vA);
            for vB = vrange
                lamB = lambda(EB, vB);
                muB  = mu(EB, vB);
                kB   = kappa(EB, vB);
                % Check material pair. We can't have A == B because EB > EA.
                if ((kA > kB) || (muA > muB))
                    continue;
                end
                for p = 1:3
                    try
                        [AStars, params] = ...
                           sequentialLaminates(lamA, muA, lamB, muB, p, 10, 10);
                        outname = sprintf('h%f_%f_%f_%i.mat', vA, EB, vB, p);
                        parsave(outname, AStars, params, EA, vA, EB, vB, p); 
                    catch err
                        disp(sprintf('Error homogenizing (lamA, muA, lamB, muB) = (%f, %f, %f, %f)', ...
                                     lambda(EA, vA), mu(EA, vA), lambda(EB, vB), mu(EB, vB)));
                        disp(err.message);
                    end
                end
            end
        end
    end
    matlabpool close;
end

% Matlab complains about a "Transparency violation error" unless this hack is
% used.
function parsave(outname, AStars, params, EA, vA, EB, vB, p)
    save(outname, 'AStars', 'params', 'EA', 'vA', 'EB', 'vB', 'p');
end
