# Frozen spatial candidate opportunity

Status at plan freeze: SOURCE_PENDING; no GPU run or result yet.

The full preview fit still selects poor global maxima: among239 native main-outside states, local centre coverage is25 and wide coverage55. Shrinking the initial template descriptor improves some outside cases but worsens overall coverage. This experiment keeps template64 and the fitted CNN unchanged and tests whether a small set of spatially separated maxima has useful geometric opportunity.

Candidate collection reads only the current auxiliary256 preview, the initial template64 central4x4 descriptor, image validity and the original tracking search size. Greedy cosine maxima suppress centres less than half the local ROI side apart. Report prefixes of at most1/2/4/8 proposals; if the finite valid grid is exhausted, the actual candidate count is recorded. Top1 must reproduce the previous fitted proposals. The query ROI sides and old auxiliary role stay fixed.

The1752 reference states are from earlier training trajectories, not new closed-loop states. GT is read after proposal selection to check centre membership and the maximum GT-area coverage of a **single** candidate ROI. This oracle diagnosis evaluates the available actions; it is neither deployable selection nor combined coverage of several queries. No heavy auxiliary encoder or gain controller executes. Candidate points and complete masked score maps are private, allowing later CPU reconstruction of greedy selection and coverage.

Admission requires a fresh independent SOURCE review and a real GPU1 frozen two-state probe with strict reload. Formal execution then covers all1752 states, with265 state tensors unchanged, no gradient/optimizer update/backward, original source32 and inputs preserved. Seed2027; use the warm environment. Physical GPU0's original full training and scheduled observer are untouched.

No change to the deployable model or final C is authorized by the diagnostic itself. If the candidate set offers useful opportunity, a subsequent separately reviewed change may train the gain head to choose a single ROI before heavy auxiliary encoding. Quality/gain supervision and dev calibration must be refreshed after the fitted preview changed. Full97-dev/245-test metrics, real system cost and all three core controls remain outstanding.
