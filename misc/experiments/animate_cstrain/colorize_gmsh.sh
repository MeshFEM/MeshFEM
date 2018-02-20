if [[ $# -ne 2 ]]; then
    echo "Usage: colorize_gmsh.sh input.png output.png"
fi
convert $1 -fill purple +opaque white $2
