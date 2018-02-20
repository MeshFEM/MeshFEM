################################################################################
# Computes explicit integrals over the canonical bilinear element of the
# matrices in 2DStiffness.mpl
################################################################################
$include "2DStiffness.mpl"
# Note: must multiply by the jacobian determinant to get the integral over a
# particular element. Conveniently, this determinant is constant for the
# axis-aligned stretches to which we restrict ourselves.
IBmat := map(int, map(int, Bmat, x=0..1), y=0..1);
IKmat := map(int, map(int, Kmat, x=0..1), y=0..1);
C(IBmat);
C(IKmat);

# Multiply by the jacobian determinant for an axis-aligned rectangular element
# with dimentions (w, h)
IKElMat := simplify(subs(invW = 1/w, invH = 1/h, evalm(w*h*IKmat)));

# Formatting regexes to run on generated code:
#   %s/0.\([0-9]\)e1\>/\1.0/g
#   %s/1.0 \/ w \/ h/invVol/g
#   %s/h \* h/hSq/g
#   %s/w \* w/wSq/g
#   %s/IKElMat\[\([0-9]\)\]\[\([0-9]\)\]/fullCellResult(\1, \2)/ 
C(IKElMat);
