# Host allocator warm source review

**SOURCE_PASS — 0 blocking issues.** New aggregate and case guards bind 55 and 52 source files respectively.

The warm wrapper forwards the original five step arguments unchanged, waits for the original step function to return, then increments its private count. It calls gc/trim exactly once at count4 and returns the original result. It forwards all eight trajectory arguments unchanged. The source preserves the original backward/optimizer step, formulas, RNG, model, data and native argv path.

A focused exact-main AST fixture uses only in-memory mocks. Six calls preserve five argument identities and six distinct return identities. Steps1-3 do not trim; the callback runs once immediately after original step4; steps5-6 do not repeat it. The memory receipt records optimizer_updates4. No real gc, libc, torch, trainer or subprocess executes in this fixture.

All 55 aggregate and 52 case source hashes match. Original 50 sources, 24 input bindings, sequence order and recipe remain unchanged. The warm controller differs from the prior trim controller only in report/status names, retaining the same finite optimizer tolerances, native0/reload/count checks, deadline and owned-group cleanup. Both reachable generated programs compile. The completed default R4 reference is hash-bound and explicitly reused without rerun.

The 16 owner records match the local measurement_start snapshot exactly, four per GPU. Live ownership/capacity is not queried by this review; the execution-time gates remain mandatory. No existing post-load wrapper, phase4 source/index or active task is edited.

The archived post-load receipt shows only 82KiB net PSS reduction before trajectory updates. This does not establish that later warm trimming helps. Warm memory benefit remains untested. The planned comparison also retains compact allocator settings and reuses an earlier reference under different concurrency; it cannot by itself establish isolated warm-trim benefit or same-load speed equivalence. R4 exact-policy failure and prior non-bit-exact optimizer records remain preserved.

This PASS is for one new short probe only. Actual native0/reload, exact model/CSV checks, finite optimizer tolerance and warm memory results remain pending. No fulltraining/accuracy/FPS improvement or new formal admission is claimed.

Requested reviewer: gpt-6-astra / max; actual model/effort: UNATTESTED. Same-family, provisional. No remote/active-job/GPU/weight/NN query or real gc/libc call occurred. Source snapshots, owner evidence, generated programs, fixture and lineage are retained in the trace directory.
