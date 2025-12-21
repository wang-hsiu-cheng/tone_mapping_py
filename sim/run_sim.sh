## Test golden pattern load ##
vcs -R +v2k -full64 -f sim.f -debug_acc+all -l vcs_test.log \
+define+LOG_LUN+PAT_L=0+PAT_U=0\
+define+FLAG_VERBOSE=1 \
+define+FLAG_DUMPWV=1 \
+define+END_CYCLES=2000000

## part0: PATCHEMBED ###
# vcs -R +v2k -full64 -f sim.f -debug_acc+all -l vcs_part0.log \
# +define+PATCHEMBED+PAT_L=0+PAT_U=0\
# +define+FLAG_VERBOSE=1 \
# +define+FLAG_DUMPWV=1 \
# +define+END_CYCLES=250000

## you can get part1 score if  finish head0 of QKV projection.

## part1_1: LNORM1 ###
# vcs -R +v2k -full64 -f sim.f -debug_acc+all -l vcs_part1_1.log \
# +define+LNORM1+PAT_L=0+PAT_U=99 \
# +define+FLAG_VERBOSE=1 \
# +define+FLAG_DUMPWV=1 \
# +define+END_CYCLES=250000

## part1_2: QKV_HEAD0 ###
# vcs -R +v2k -full64 -f sim.f -debug_acc+all -l vcs_part1_2.log \
# +define+QKV_HEAD0+PAT_L=0+PAT_U=0 \
# +define+FLAG_VERBOSE=1 \
# +define+FLAG_DUMPWV=0 \
# +define+END_CYCLES=250000

# vcs -R +v2k -full64 -f sim.f -debug_acc+all -l vcs_part1_2.log \
# +define+QKV_HEAD1+PAT_L=0+PAT_U=99 \
# +define+FLAG_VERBOSE=1 \
# +define+FLAG_DUMPWV=1 \
# +define+END_CYCLES=250000

## you can get part2 score if  finish head0 and chunk0 of QK multiplication and softmax.
## you can check the correctness of other heads and chunks by writing them to SRAM.

## part2_1: QKMUL_HEAD0_CHUNK0 ###
# vcs -R +v2k -full64 -f sim.f -debug_acc+all -l vcs_part2_1.log \
# +define+QKMUL_HEAD0_CHUNK0+PAT_L=0+PAT_U=0 \
# +define+FLAG_VERBOSE=1 \
# +define+FLAG_DUMPWV=1 \
# +define+END_CYCLES=250000

# vcs -R +v2k -full64 -f sim.f -debug_acc+all -l vcs_part2_1.log \
# +define+QKMUL_HEAD1_CHUNK3+PAT_L=0+PAT_U=0 \
# +define+FLAG_VERBOSE=1 \
# +define+FLAG_DUMPWV=0 \
# +define+END_CYCLES=250000

## part2_2: SOFTMAX_HEAD0_CHUNK0 ###
# vcs -R +v2k -full64 -f sim.f -debug_acc+all -l vcs_part2_2.log \
# +define+SOFTMAX_HEAD1_CHUNK0+PAT_L=0+PAT_U=99 \
# +define+FLAG_VERBOSE=0 \
# +define+FLAG_DUMPWV=0 \
# +define+END_CYCLES=250000

## finish all attention heads and chunks to get part3 score.

## part3: VMUL_HEAD0(two heads) ###
# vcs -R +v2k -full64 -f sim.f -debug_acc+all -l vcs_part3.log \
# +define+VMUL_HEAD1+PAT_L=0+PAT_U=99 \
# +define+FLAG_VERBOSE=0 \
# +define+FLAG_DUMPWV=0 \
# +define+END_CYCLES=1000000

# part4_1: OPROJ ###
# vcs -R +v2k -full64 -f sim.f -debug_acc+all -l vcs_part4_1.log \
# +define+OPROJ+PAT_L=0+PAT_U=99 \
# +define+FLAG_VERBOSE=0 \
# +define+FLAG_DUMPWV=0 \
# +define+END_CYCLES=1000000

# part4_2: RESIDUAL1 ###
# vcs -R +v2k -full64 -f sim.f -debug_acc+all -l vcs_part4_2.log \
# +define+RESIDUAL1+PAT_L=0+PAT_U=99 \
# +define+FLAG_VERBOSE=0 \
# +define+FLAG_DUMPWV=0 \
# +define+END_CYCLES=1000000

### Notice1: set FLAG_VERBOSE to 1 for detailed simulation reports
### Notice2: set FLAG_DUMPWV to 1 for dumping the fsdb waveform
