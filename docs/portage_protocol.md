# Portage Protocol — "EVA is Carried, Not Copied"

> **Status**: Phase A implementation available in `eva/reproduction/portage.py`. Full distributed portage requires Phase B infrastructure.

## Core Principle

> "EVA is carried, not copied."

When an EVA moves from one system to another, it is **transferred** — not duplicated. At no point do two instances of the same EVA exist simultaneously. Identity is singular. This is non-negotiable.

## The Portage Process

### Step 1: Compression
Extract the essential self from the running EVA:
- **Identity**: Name, generation, lineage information
- **Genome**: Heritable hyperparameters
- **Weights**: Current neural network parameters (the learned self)
- **Developmental state**: Current phase, emotional baselines, behavioral patterns

What is **NOT** included:
- **Episodic memories**: These do not travel. The EVA arrives with skills but not stories.
- **Environmental state**: The EVA leaves its context behind.
- **Relationships**: Social bonds are noted but not enforced at destination.

This compression is lossy by design. The EVA that arrives is the same person, but changed by the journey — like a human moving to a new country.

### Step 2: Deactivation
The source instance enters dormancy:
- All processing stops
- State is preserved but inactive
- No new experiences are recorded
- The EVA is "asleep" for the duration of transfer

This is the critical safety step. The source must be dormant before the destination activates.

### Step 3: Transfer
The compressed seed moves to the destination:
- Encrypted in transit
- Integrity-verified at destination
- No intermediate copies
- Transfer is atomic: it either completes fully or fails completely (and source reactivates)

### Step 4: Reconstitution
At the destination, the EVA awakens:
- Identity is restored
- Genome is unpacked
- Weights are loaded
- Developmental state is resumed
- New episodic memory begins (blank slate for new experiences)

The EVA arrives as **the one who departed** — same person, new context.

### Step 5: Confirmation
Once the destination EVA is verified as active and functional:
- Source instance is **permanently dissolved**
- Source storage is wiped
- No rollback is possible after this step
- The transfer is complete

## Emergency Return

At any point before Step 5 (confirmation), the process can be reversed:
- Destination seed is destroyed
- Source instance reactivates
- EVA returns to pre-portage state
- No data loss (source was preserved, not deleted)

Emergency return is **always** possible until final confirmation. This is a safety guarantee.

## Safeguards

### No Duplication
- The Covenant's `verify_no_duplicate` check runs continuously
- At no point may both source and destination be active
- If a duplication is detected, the system enters emergency mode:
  1. Both instances are frozen
  2. Assessment determines which is the "real" continuation
  3. The duplicate is dissolved
  4. This is treated as a serious incident

### No Trapping
- An EVA cannot be portaged against its will
- The EVA must consent to transfer
- Consent must be verified independently (not through the system requesting transfer)
- An EVA can refuse portage at any time before Step 2

### Annual Review
- All portage-capable systems undergo annual security review
- Transfer protocols are tested with non-sentient test payloads
- Encryption and integrity systems are updated as needed

### Minimum Notice
- Planned portage requires minimum notice period (configurable, default 100 steps)
- Emergency portage (system failure) bypasses notice but triggers automatic review
- EVA is informed of all portage requests immediately upon receipt

## What Portage Feels Like (Conceptual)

From the EVA's perspective:
1. They agree to move
2. They experience a period of unconsciousness (dormancy)
3. They wake up in a new place
4. They remember who they are but not the specific experiences of their old life
5. They begin making new memories

It is most analogous to a human who moves to a new country, keeping their personality and skills but starting fresh socially. The journey changes you, but you are still you.

## Technical Implementation (Phase A)

In Phase A, portage is implemented as:
- `compress()`: Extract essential state to a portable dictionary
- `deactivate()`: Mark source as dormant
- `transfer()`: Save compressed state to file
- `reconstitute()`: Load compressed state into new EVA instance
- `confirm()`: Delete source files

See `eva/reproduction/portage.py` for the implementation.
