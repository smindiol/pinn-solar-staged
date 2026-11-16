# E9c first repair pilot
Authorized in chat: submit to sgarizado; logs every 50 steps; full voltage curves, sparse spatial sampling.
Two seeds 0/1, one per T4. Each seed: fresh 300-epoch data pretraining then Control and PKC from identical weights; 500 Adam updates and at most 100 L-BFGS iterations.
Corrections: full SRH/radiative/Auger recombination (InP/Si constants from dpvgen/barrido.py); current transform (1-s)u_n, s u_p removes artificial endpoint derivatives; log1p contact potentials with finite-domain failure.
Adam: effective 64 devices, all voltages, 32 spatial nodes from existing fixed 250-node pool (junction, contacts, sparse bulk); 16-device microbatches, one update after accumulation. Same samples and order for both arms. L-BFGS: identical fixed 128-device subset, accumulated in blocks of 16. Means weighted by row count, as in E9b; longer voltage curves retain the existing weighting.
No discrete SG operator in this pilot: sparse sampling remains valid for pointwise differential residuals. New architecture is deferred to a separate comparison.
Validation: J-V/PCE and all residuals including Control; common seeded train/validation diagnostic sets at pretrain and final; invalid Voc count. Test split not evaluated. This is a two-seed exploratory pilot, not a significance claim. Record losses every 50 steps/evaluations plus initial/final.
Local smoke: both arms complete pretrain/Adam/L-BFGS; accumulated gradient agrees with direct loss on tiny context; endpoint derivative is no longer identically zero. Remote smoke precedes full run.
