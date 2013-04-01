with(CodeGeneration):
with(LinearAlgebra):
# invH and invW terms are from chain rule!
p0x, p1x, p2x, p3x := invW * (y - 1),  invW * (1 - y), invW * y,      -invW * y:
p0y, p1y, p2y, p3y := invH * (x - 1), -invH *       x, invH * x, invH * (1 - x):
Bmat := << p0x |   0 | p1x |   0 | p2x |   0 | p3x |   0 >,
         <   0 | p0y |   0 | p1y |   0 | p2y |   0 | p3y >,
         < p0y | p0x | p1y | p1x | p2y | p2x | p3y | p3x >>;
Dmat := << d00 | d01 |   0>,
         < d10 | d11 |   0>,
         <   0 |   0 | d22>>;
Kmat := map(expand,evalm(Transpose(Bmat) &* Dmat &* Bmat));
C(Kmat);
