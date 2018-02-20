################################################################################
# Generates a point's contribution to the per-element laplacian matrix for
# bilinear quad elements. (x, y) are the point's coordinates in the canonical
# reference quad.
# "Lmat":                  (per-point value to be integrated over the element)
# L_e = \int_e LMat dV     (per-element laplacian matrix)
# L = \sum_e unpack(L_e)   (full laplacian matrix; unpack expands 4x4
#                           per-element matrix into an n x n matrix)
# We want \int_e Lmat u_e dV to compute the laplacian of u_e. To do this, we
# have Lmat u_e compute:  [(\grad u_e)^T (\grad \phi_0),
#                                      ...
#                          (\grad u_e)^T (\grad \phi_3)]
# So Lmat is a matrix of shape function gradient dot products:  
#      [ (\grad \phi_0)^T grad \phi_0,  ..., (\grad \phi_3)^T grad \phi_0]
#      [           ...                  ...                ...           ]
#      [ (\grad \phi_0)^T grad \phi_3,  ..., (\grad \phi_3)^T grad \phi_3]
# Which can be computed as B^T * B if B is a matrix of shape function gradient
# column vectors.
###############################################################################
with(CodeGeneration):
with(LinearAlgebra):
# Shape function partial derivatives
# invH and invW terms are from chain rule!
p0x, p1x, p2x, p3x := invW * (y - 1),  invW * (1 - y), invW * y,      -invW * y:
p0y, p1y, p2y, p3y := invH * (x - 1), -invH *       x, invH * x, invH * (1 - x):
Bmat := << p0x |  p1x | p2x | p3x >,
         < p0y |  p1y | p2y | p3y >>;
Lmat := map(expand,evalm(Transpose(Bmat) &* Bmat));
C(Lmat);
