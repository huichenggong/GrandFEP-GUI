base=$PWD
for system in 3RKZ_lig62to70_alpha05 MHT1_lig3_alpha05 2B8V_lig24and25_alpha05 ck2_custcore_hotlys 2E9P_lig4to7_alpha05 2Q15_lig17to21_alpha05 hsp90_3hvd_custcore 
do
    cd $base/
    /home/cheng/E29Project-2026-03-01/155-GrandFEP-GUI/benchmark/mapping_test/public_binding_free_energy_benchmark/fep_benchmark_inputs/mapping_script/map.py \
    --csv ${system}_edges.csv \
    --sdf ${system}_ligands.sdf -o $system
    cd $base/$system
    # ../../../../../../Script/parameterize_sdf_antechamber.py ../${system}_ligands.sdf -o ligand_preparation 
    ../../../mapping_script/check_constraint.py ./
done
