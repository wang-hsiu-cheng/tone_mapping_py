## Test golden pattern for Log Luminance
# vcs -R +v2k -full64 -f sim.f -debug_acc+all -l vcs_test_0.log \
# +define+LOG_LUN+PAT_L=0+PAT_U=9\
# +define+FLAG_VERBOSE=0 \
# +define+FLAG_DUMPWV=0 \
# +define+END_CYCLES=30000000

## Test golden pattern for Base Layer
# vcs -R +v2k -full64 -f sim.f -debug_acc+all -l vcs_test_1.log \
# +define+BASE_LAYER+PAT_L=0+PAT_U=9\
# +define+FLAG_VERBOSE=0 \
# +define+FLAG_DUMPWV=0 \
# +define+END_CYCLES=30000000

## Test golden pattern for LDR Output RGB
vcs -R +v2k -full64 -f sim.f -debug_acc+all -l vcs_test_2.log \
+define+LDR+PAT_L=0+PAT_U=9\
+define+FLAG_VERBOSE=0 \
+define+FLAG_DUMPWV=0 \
+define+END_CYCLES=30000000
