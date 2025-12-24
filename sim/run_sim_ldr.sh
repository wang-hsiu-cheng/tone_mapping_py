## Test golden pattern load ##
# vcs -R +v2k -full64 -f sim.f -debug_acc+all -l vcs_test.log \
# +define+LOG_LUN+PAT_L=0+PAT_U=0\
# +define+FLAG_VERBOSE=1 \
# +define+FLAG_DUMPWV=1 \
# +define+END_CYCLES=2000000

## Test golden pattern load ##
vcs -R +v2k -full64 -f sim_ldr.f -debug_acc+all -l vcs_test.log \
+define+BASE_LAYER+PAT_L=0+PAT_U=0\
+define+FLAG_VERBOSE=1 \
+define+FLAG_DUMPWV=1 \
+define+END_CYCLES=8000000