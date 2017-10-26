for i in $(cat validation_patterns.txt); do
    result=$SCRATCH/nico_patterns/$i.wire.isovalidate.txt
    aniso=$(grep Anisotropy $result | cut -f2)
    relErrorCompliance=$(grep "Rel error compliance 98th percentile" $result | cut -f2)
    transRelErrorCompliance=$(grep "Transformed rel error compliance 98th percentile" $result | cut -f2)
    if [[ -e $result ]]; then
        echo fail;
    fi
done
