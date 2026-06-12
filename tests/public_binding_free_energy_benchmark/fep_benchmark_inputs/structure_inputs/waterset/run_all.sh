base=$PWD
for system in brd41_ASH106 chk1 hsp90_kung hsp90_woodhead scyt_dehyd taf12 throm_nozob_hip75 urokinase
do
    cd $base/$system
    # ../../../../../../Script/parameterize_sdf_antechamber.py ../${system}_ligands.sdf -o ligand_preparation 
    cp -r /home/cheng/E29Project-2026-03-01/155-GrandFEP-GUI/benchmark/mapping_test/public_binding_free_energy_benchmark/fep_benchmark_inputs/structure_inputs/waterset/$system/* ./
done