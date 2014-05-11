################################################################################
# Generates a point's contribution to the per-element linear elasticity
# stiffness matrix for bilinear quad elements. (x, y) are the point's
# coordinates in the canonical reference quad.
#  "Kmat":                 (per-point value to be integrated over the element)
#  K_e = \int_e Kmat dV    (per-element stiffness matrix)
#  K = \sum_e unpack(K_e)  (full stiffness matrix; unpack expands 8x8
#                           per-element matrix into a 3n x 3n matrix)
#  We want u_e^T Kmat u_e to compute <epsilon, sigma> where epsilon and sigma are
#  2x2 symmetric strain and strain matrices at the point. These are stored in
#  "flattened" form [e_00, e_11, e_12]. We can compute the matrix inner
#  product by using an inner product on the flattened vectors if we make the
#  off-diagonal contribute twice:
#      <epsilon, sigma> = [e_00, e_11, 2 * e_12]^T [s_00, s_11, s_12]
#  To do this, we have B u actually compute "engineering strain:"
#  ([e_00, e_11, 2 * e_12]), and then "D" is the material matrix with
#  the ***last row halved***. This means epsilon is engineering strain and
#  sigma is plain stress (not engineering stress). Now u^T B^T D B u correctly
#  computes the inner product <epsilon, sigma>.
################################################################################
with(CodeGeneration):
with(LinearAlgebra):
# Shape function partial derivatives
# invH and invW terms are from chain rule!
p0x, p1x, p2x, p3x := invW * (y - 1),  invW * (1 - y), invW * y,      -invW * y:
p0y, p1y, p2y, p3y := invH * (x - 1), -invH *       x, invH * x, invH * (1 - x):
Bmat := << p0x |   0 | p1x |   0 | p2x |   0 | p3x |   0 >,
         <   0 | p0y |   0 | p1y |   0 | p2y |   0 | p3y >,
         < p0y | p0x | p1y | p1x | p2y | p2x | p3y | p3x >>;
# Note: expects d22 to be half the actual material matrix's lower right value
Dmat := << d00 | d01 |   0>,
         < d10 | d11 |   0>,
         <   0 |   0 | d22>>;
Kmat := map(expand,evalm(Transpose(Bmat) &* Dmat &* Bmat));
C(Kmat);
