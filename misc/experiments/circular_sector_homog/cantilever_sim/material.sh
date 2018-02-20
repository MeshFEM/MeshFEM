poisson=$1
cat <<END
{
    "type": "isotropic_material",
    "dim": 3,
    "density": 1.0,
    "young": 200.0,
    "poisson": $poisson
}
END
