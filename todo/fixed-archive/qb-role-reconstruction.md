### [FIXED] QB availability rewrite removed the intended historical role proxy

**File(s):** `src/features/engineer.py`, `src/serving/upcoming_week.py`,
`tests/test_availability_population.py` (PR pending).

**What:** #1564 replaced historical QB participation-based availability with
active-roster membership. That removed an intentional approximation for sparse
historical role/injury feeds. The retained prior-opportunity ranking could then put
an active backup ahead of the expected starter; merely restoring proxy training
left a mismatch when upcoming prediction still used the roster-only rule.

**Fix:** Historical QB rows reconstruct the proxy with corrected inputs and
out-set semantics. Upcoming rows select eligible QBs by available pregame depth,
using prior opportunity for ties, fallback, and vacancy magnitude. The live builder
supplies depth before inheritance is calculated. Other positions preserve their
existing behavior. Production-pipeline comparisons and information-boundary checks
are recorded in [the validation report](../qb-role-reconstruction-validation.md).

**Lesson:** Distinguish a historical information-reconstruction assumption from
an as-of source. Removing a useful proxy is a model-contract change. Validate both
the historical training representation and its prediction-time reconstruction;
an active roster does not by itself establish expected playing role.
