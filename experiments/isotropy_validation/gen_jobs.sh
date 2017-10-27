for i in $(cat validation_patterns.txt); do
    run=$(readlink -f run_isotropic_validation.sh)
    create_pbs.sh validate_$i "$run $SCRATCH/nico_patterns/$i.wire" 2 8 0 15 | tee $i.pbs
done
