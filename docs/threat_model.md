# AURA threat model

What AURA's results claim to cover, and what they do not. Every attack named here maps to a
script and a results file.

## System

- **Members:** five organisations (hospital, bank, university, ISP, retail). Each sees only the
  NetFlow records whose destination IP is in its network.
- **Server:** coordinates rounds, holds a small clean root dataset, aggregates autoencoder (AE)
  updates with FLTrust or DC-FLTrust (`aura/fl_server.py`, round recipe in `aura/fl_round.py`).
- **Starting model (G0):** trained centrally on the training pool before federation starts.
- **Local only:** each organisation's graph model (GraphSAGE) and its raw flows never leave it.

## What is protected

1. **Detection quality:** recall at 1% false positives on benign traffic, at every member.
2. **Members' data:** raw flows, and properties of them, such as whether a member is under attack.
3. **Integrity of the update record:** which update each member sent in each round (Merkle log).

## Parties and what they can do

| Party | Trusted for | Capabilities assumed |
|---|---|---|
| Honest member | Following the protocol | May hold attack traffic in its own data (it is a victim, not an attacker) |
| Malicious member | Nothing | Sends any update it likes; may train on attack traffic; may know its own honest update; up to 2 of 5 colluding |
| Root-aware malicious member | Nothing | As above, and knows the server's root update (worst case: holds the root data) |
| Server | Aggregating correctly; holding clean root data | Sees every individual update and every trust score (honest but curious) |
| Network attacker | Nothing | Can modify updates in transit (Wave 6, Merkle log) |

## Attacks tested (development holdout; headline cells also on the test split once)

| Attack | Attacker goal | Script | Results |
|---|---|---|---|
| Sign flip (x1), 1 or 2 attackers | Undo learning | `scripts/wave3_byzantine.py` | `results/wave3_byzantine.json` |
| Scaled flip (x10), 1 attacker | Overwhelm the average | same | same |
| Data poisoning, 1 or 2 | Make the AE reconstruct attacks (evasion) | same | same |
| Adaptive poisoning, cosine 0.6-0.0 with own honest update, 2 | Look honest while poisoning | `--sweep` | `results/wave3_adaptive_sweep.json` |
| Root-aware poisoning, cosine 0.6-0.02 with the root update, 2 | Game FLTrust's score directly | `--rootaware` | `results/wave3_rootaware.json` |
| Krum with f=1 vs one attacker | Control: Krum inside its assumptions | `--krum-control` | `results/wave3_krum_control.json` |
| Head sign flip | Undo the attack head | `scripts/final_federation.py` | `results/final_federation.json` |
| None (self-filtering cut) | No attacker: does the system blind itself? | `scripts/ae_routing_cut.py`, `scripts/cut_quantile_rule.py` | `results/ae_routing_cut.json`, `results/cut_quantile_rule*.json` |
| Honest-but-curious server | Infer which member is under attack | `scripts/trust_leak.py` | `results/trust_leak.json` |
| Gradient leakage (DLG), membership inference | Reconstruct or identify flows | Wave 4 (not yet rerun since the fixes) | stale: `results/dlg_test.json` |

## Out of scope, and why it matters

- **More than 2 of 5 malicious members.** FLTrust has no bound here; Krum needs n > 2f + 2, so it
  is out of spec at f = 2 with 5 members.
- **Multi-round attacks** (PoisonedFL-style consistency, Faker similarity gaming). Untested; each
  round's attack is crafted independently.
- **Poisoned root data.** FLTrust trusts the root completely; a poisoned root breaks it.
- **Sybils** (one party joining as several members). Membership is assumed fixed.
- **Attacks on G0's central training.** G0 is assumed clean.
- **Evasion at inference only** (adversarial flows against a clean model), beyond what the
  dataset's attack classes already contain.
- **Secure aggregation.** Not used, and incompatible with FLTrust, which must see each update to
  score it. That is why the trust score can leak incident status.

## Assumptions that the results depend on

- Flows are shuffled draws per member, so there is no daily drift; attack shares are fixed per run.
- The alarm threshold is the benign 99th percentile on calibration flows, computed by the server.
- Federation runs in-process; the networked Flower path implements the same round recipe but has
  not been rerun since the fixes.
