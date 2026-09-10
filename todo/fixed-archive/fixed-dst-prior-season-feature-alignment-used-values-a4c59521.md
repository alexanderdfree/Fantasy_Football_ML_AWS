> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] DST prior-season feature alignment used `.values`
- **File:** `src/dst/features.py:74-86`
- **What:** Prior-season features were merged via `season+1` then assigned back using `.values` (strips index). If the merge reordered rows, assignments would be silently misaligned.
- **Fix:** Changed to index-preserving merge: `reset_index()` → merge → `set_index("index")` → loc-based assignment using the original index.
