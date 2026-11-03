# P1B Full Collection to FIT

The local operator `work/prepare_p1b_completed_fit.py` now connects the
already deployed canonical full verifier to the existing disabled scaled
FIT preparer. Run it from the repository using the existing local Python:

```powershell
E:/anaconda/python.exe -B work/prepare_p1b_completed_fit.py dropmoe
E:/anaconda/python.exe -B work/prepare_p1b_completed_fit.py sot
```

Each branch uses its original snapshot, absolute NN interpreter and four
original ordered collection plans. Before invoking the verifier it checks
the deployed verifier/review hashes and four exact native commands.
Incomplete branches print `NOT_READY` and their native states without
creating acceptance, drafts or GPU tasks. Both real hosts exercised this
pending path successfully on 2026-11-03 around 04:52 UTC.

Once all four original native tasks are COMPLETE/exit0, the command calls
the already admitted CPU verifier once, requiring 882 sequences,
458,553 tracked frames, 91,091 records and CUDA uninitialized. It archives
the exact acceptance/native bytes and verifier output under
`reports/p1b_full_fit_r1_20261102/<branch>/`.
The successful acceptance path has not yet executed as of this note.
Do not create this output directory manually before running the command.
Existing acceptance/output paths cause an assertion; preserve them and
inspect their evidence instead of overwriting or rerunning the verifier.

The local disabled draft goes in that branch's `draft/` subdirectory:
GPU3, B512, four loader workers, five epochs, preview LR4e-5 and
quality/gain LR4e-4. Task names are
`fit_p1b_<dropmoe,sot>_full_b512_r1_p<25,26>_20261102`.
No draft is deployed and no FIT is launched by this operator.

After real acceptance and draft creation, request a separate exact
concrete plan/input/source/recipe review in the already authorized review
chat. Source-only scaled-trainer review is not concrete admission.
Check actual report stage, original98 source map, supplemental trainer
hash and canonical plan binding. Then deploy only the admitted new files,
run canonical CPU admission, check fresh GPU3 capacity and launch once
through the existing native wrapper. B512 two-update capacity evidence
already exists; do not rerun capacity experiments.

Actual FIT must preserve executor state, update all three policy modules,
retain all five epoch checkpoints, pass strict reload and finish natively.
Its TRAIN loss and duration must be measured from the actual run.
Repeated-batch capacity timing is only a rough estimate, not convergence
or full-loader timing. Round2 remains dependent on real round1 completion
and bound epoch005 behavior followed by fresh TRAIN trajectories.
