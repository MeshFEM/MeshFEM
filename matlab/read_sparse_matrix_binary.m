function A = read_sparse_matrix_binary(path)
% Read sparse matrix in binary triplet format.
fd = fopen(path, 'r');
n = fread(fd, 1, 'uint64');
i = fread(fd, n, 'uint64');
j = fread(fd, n, 'uint64');
v = fread(fd, n, 'double');
i = i + 1;
j = j + 1;
A = sparse(i, j, v);
end
