#!/bin/zsh
# Mac OS 10.11 and later clear the DYLD flags in the name of security.
export DYLD_FALLBACK_LIBRARY_PATH=$DYLD_LIBRARY_PATH:$BYPASS_SIP_DYLD_PATH
for i in {0..4}; do
    ../../tools/grid $((5 * 2**$i))x$((2**$i))x$((2**$i)) -t bar_tet_$i.msh
done
