base=$PWD
for dir in $(ls -d A??); do
    echo "Processing directory: $dir"
    cd $base/$dir
    cat > "$base/$dir/01_dry_tleap.in" <<EOF
source leaprc.water.tip3p
source leaprc.gaff2

MOL = loadmol2 $dir.mol2
loadamberparams $dir.frcmod

set MOL box { 30.0 30.0 30.0 }

savepdb       MOL 01_dry.pdb
saveamberparm MOL 01_dry.prmtop 01_dry.inpcrd

quit
EOF
    tleap -f 01_dry_tleap.in
done
