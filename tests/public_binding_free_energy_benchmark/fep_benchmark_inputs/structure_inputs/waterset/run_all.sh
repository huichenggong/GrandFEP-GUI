base=$PWD
for system in brd41_ASH106 chk1 hsp90_kung hsp90_woodhead scyt_dehyd taf12 throm_nozob_hip75 urokinase
do
    cd $base/$system
    # ../../../../../../Script/parameterize_sdf_antechamber.py ../${system}_ligands.sdf -o ligand_preparation 
    ../../../mapping_script/check_constraint.py ./
done