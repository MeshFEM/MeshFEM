mesh=$1
square_mesh=$2
gmsh -n $mesh.msh render.opt
for i in frame_*.png; do
    ./colorize_gmsh.sh $i $mesh.$i
done
gmsh -n $square_mesh.msh render.opt

for i in frame_*.png; do
    convert \( $i -fill LightGray +opaque white \) \( $mesh.$i -transparent white \) -composite \
            -resize 640x640 -gravity center -extent 640x640 composite.$i
done
yes | ffmpeg -framerate 30 -i composite.frame_%03d.png -vcodec libx264 -pix_fmt yuv420p -crf 18 $mesh.mp4
