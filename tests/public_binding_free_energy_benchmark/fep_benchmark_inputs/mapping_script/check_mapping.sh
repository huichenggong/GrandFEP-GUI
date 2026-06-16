dataset_system=$1
script_dir=$PWD

if [ -z "$dataset_system" ]; then
  echo "Usage: $0 <system.sdf>"
  exit 1
fi

dataset=$( dirname "$dataset_system" )
system_sdf=$( basename "$dataset_system" .sdf )
# remove _ligands from system name
system=${system_sdf%%_ligands}
echo "Dataset    : $dataset"
echo "System sdf : $system_sdf"
echo "System     : $system"

# exit 0

cd ../structure_inputs/$dataset/

if [ ! -d "$system" ]; then
    echo "  Mapping output directory $system does not exist for system $system"
    continue
fi
cd $system
echo "Here are the edges :"
ls -d edge_*
base=$PWD
for edge_dir in $( ls -d edge_* ); do
    cd $base/${edge_dir}
    if [ ! -f "mapping_visial_checked.json" ]; then
        cp mapping_constraint_checked.json mapping_visial_checked.json
    fi
    /home/cheng/Software/miniforge3/envs/pymol/bin/pymol -r $script_dir/view_edge.py -- \
        ./mol_a.sdf \
        ./mol_b.sdf \
        ./mapping_visial_checked.json
    sleep 0.1
done
