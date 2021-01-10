mkdir ../ALL_output

python3 train_arch.py \
--run_mode normal \
--N_epoch_EDTG 600 \
--loss Hinge

mv ../output ../ALL_output/output_Hinge