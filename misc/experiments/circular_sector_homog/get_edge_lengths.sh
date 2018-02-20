#!/usr/bin/env zsh
for i in mesh_*.msh; do echo -e ${${i%.msh}#mesh_}\\t$(mesh_convert -i $i | grep length | cut -f2 | tr '\n' '\t'); done
