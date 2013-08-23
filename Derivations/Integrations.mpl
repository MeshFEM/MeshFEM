################################################################################
# Computes explicit integrals over the canonical bilinear element of the
# matrices in 2DStiffness.mpl
################################################################################
$include "2DStiffness.mpl"
IBmat := map(int, map(int, Bmat, x=0..1), y=0..1);
IKmat := map(int, map(int, Kmat, x=0..1), y=0..1);
C(IBmat);
C(IKmat);
