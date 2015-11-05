#!/usr/bin/env zsh
echo "flippers = ["
for bc in results/*(/); do
    bcname=$(basename $bc)
    sed "s/<bcond>/$bcname/" frames_template.js >! $bc/frames.js
    echo "['bc $bcname', '$bcname/frames.js'],"
done
echo "];"
