echo "flippers = ["
for solver in {MeshFEM,Mathematica}; do
for i in results/skip_0/poisson_*; do
    poisson=${i#*poisson_}
    sprefix=""
    [[ $solver == "Mathematica" ]] && sprefix="mathematica."
    sed "s/<nu>/$poisson/g; s/<sprefix>/$sprefix/g" frames_template.js > results/${solver}_poisson_$poisson.js
    echo "\t['nu $poisson:$solver','${solver}_poisson_$poisson.js'],"
done
done
echo "];"
