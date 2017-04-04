# Ran with revision r73.
for i in {flapmix_narrow,flapmix_moderate,flapmix_wide,flap_narrow,flap_moderate,flap_wide,half_period,full_period,bump_half,bump_third}.msh; do $MeshFEM/mesh_convert $i -s ${i%.msh}_refine.msh; done
for i in *_reference.msh; do $MeshFEM/mesh_convert $i -q2 -f ${i%reference.msh}tri.msh; done
for i in *_tri.msh; $MeshFEM/tools/msh_processor $i -e young -r E -e poisson -r nu -o ${i%.msh}.target_mat.msh
for i in *_tri.msh; $MeshFEM/tools/msh_processor $i -e fitted_young_x -r E_x -e fitted_young_y -r E_y -e fitted_poisson_yx -r nu_yx -e shear_xy -r mu -o ${i%.msh}.fit_mat.msh

for i in flap*_tri.msh; $MeshFEM/Simulate_cli -b compression_y_relative_2D.bc $i -m ${i%.msh}.target_mat.msh -o ${i%.msh}.target_sim.msh
for i in flap*_tri.msh; $MeshFEM/Simulate_cli -b compression_y_relative_2D.bc $i -m ${i%.msh}.fit_mat.msh -o ${i%.msh}.fit_sim.msh
for i in flap{,mix}_{narrow,moderate,wide}{,_refine}.msh; $MeshFEM/Simulate_cli -b compression_y_relative_2D.bc $i -m ProJet7000_2D.material -o ${i%.msh}.micro_sim.msh

for i in {half_period,full_period,bump_half,bump_third}_tri.msh; $MeshFEM/Simulate_cli -b compression_relative_2D.bc $i -m ${i%.msh}.target_mat.msh -o ${i%.msh}.target_sim.msh
for i in {half_period,full_period,bump_half,bump_third}_tri.msh; $MeshFEM/Simulate_cli -b compression_relative_2D.bc $i -m ${i%.msh}.fit_mat.msh -o ${i%.msh}.fit_sim.msh
for i in {half_period,full_period,bump_half,bump_third}{,_refine}.msh; $MeshFEM/Simulate_cli -b compression_relative_2D.bc $i -m ProJet7000_2D.material -o ${i%.msh}.micro_sim.msh

for i in *_tri.msh; do name=${i%.msh}; for t in {fit,target}; do gmsh -n $name.${t}_sim.msh render.geo; mv render.png $name.${t}_sim.png; done; done
for i in *.micro_sim.msh; do gmsh -n $i render.geo; mv render.png ${i%msh}png; done
for i in *_tri.msh; do name=${i%.msh}; gmsh -n $name.target_mat.msh render_iso.geo; for m in {E,nu}; mv $m.png $name.target_$m.png; done
for i in *_tri.msh; do name=${i%.msh}; gmsh -n $name.fit_mat.msh render_ortho.geo; for m in {E_x,E_y,nu_yx,mu}; mv $m.png $name.fit_$m.png; done
