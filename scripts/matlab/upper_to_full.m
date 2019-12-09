function Afull = upper_to_full(A)
% upper_to_full Duplicate the (strict) upper triangle of an upper triangular
% matrix into its lower triangle.
    Afull = triu(A, 1)' + A;
end
