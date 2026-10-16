# Handover: how the arm client scores task progress and when it stops a trial

October 16, 2026, deployment agent, for whoever runs or evaluates policies on the Trossen WXAI Tower of Hanoi rig.
Code: `examples/hanoi/deployment/` in the OpenPI checkout (`progress.py`, `dense_client.py`, `hardware.py`,
`dense_execution.py`). Every number below is a default that a flag can change; the flag is named where it exists.

## 1. Where the board state comes from

There is no vision of the board. The client infers the board from the policy's own gripper commands, which are
the only moments a ring can change peg:

- The arm starts from a known board: the full tower on the task's start peg (the `--task` direction names it),
  or, for a play-style goal, the tower the start board names.
- A **jaw close** is a grasp. The peg is the one whose column the commanded tool position is in
  (`peg_of`: y below -20 mm is A, below 50 mm is B, else C). The ring is the top ring of that peg's stack.
  The commanded z is recorded and compared with the four ring levels (57.1, 67.8, 77.5, 87.7 mm tool height
  for a ring resting at level 1 to 4) as `level_error_mm`.
- A **jaw open** is a release on the peg the tool is over. The held ring goes onto that peg's stack.

`BoardTracker` (`progress.py`) keeps the three stacks, bottom to top, and logs a `grasp` and a `move` event for
every pair. Each move is classed:

| kind | meaning |
|---|---|
| `optimal` | reduced the graph distance to the goal by one (on a shortest path) |
| `detour` | legal, but did not bring the goal closer |
| `null` | released on the peg it came from |
| `illegal` | a larger ring put on a smaller one |
| `empty` | closed on nothing, then opened: not a ring move |

The gripper command, not the measured jaw, decides that a ring moved; the jaw stroke check in section 3 is
what catches a close that did not actually take a ring.

## 2. Progress

Distances are shortest legal move counts in the 81-board Hanoi graph (breadth-first search; `remaining_moves`,
`shortest_path`). For a trial from board S to goal G with distance d = dist(S, G):

- `progress = (d - remaining) / d`, where `remaining` is the distance from the current board (a held ring is
  counted as still on its source peg) to G, floored at 0. For the full tower d = 15; for the arm protocol's
  shorter goals d is 1, 3 or 7.
- `peak_progress` is the best progress of any board reached during the trial. It is what counts when the
  final board is unscorable, which happens after an illegal stacking (not a Hanoi state) or after a ring was
  lost (section 3).
- `moves_before_first_error`: leading moves of kind `optimal`. An error is any other kind.
- `solved`: the board equals G, no ring is held, and the board is not marked uncertain.

All of this is in the run's `summary.json` (`progress`, `peak_progress`, `remaining_moves`, `move_counts`,
`moves_before_first_error`, `optimal_prefix`, `solved`, `final_board`, `held_at_end`, `board_uncertain`) and can
be recomputed offline from `events.jsonl` with `progress.reconstruct`. The scoreboards and the comparison chart
are built from those summaries; a trial's progress on the chart is `progress`, or `peak_progress` when the
final board is unscorable.

## 3. Stopping when the jaw closes on the rod, or on nothing

The gripper closes under force control (-20 N for 2.4 s), so after a close the measured jaw stroke is the width
of whatever is between the fingers: a ring reads 15 to 30 mm (ring 1 about 16 mm, ring 3 about 25 mm), the peg
itself or empty air reads at or below 8 mm.

While the jaw is commanded closed, every control tick (30 Hz) reads the stroke and `check_grasp` raises
`MissedGraspError` the moment it is at or below `--min-grasp-stroke-m` (default 8 mm). That covers three cases
with one rule: the fingers closed on the rod, the fingers closed beside the ring on nothing, or a held ring
slipped out during the carry. The client then:

1. logs a `missed_grasp` event with the stroke and sets the status to `missed_grasp`;
2. corrects the board: if no Cartesian segment ran since the close, the ring never left its peg and is put back
   in the tracker; if the arm had already moved, the ring's whereabouts are unknown and the board is marked
   uncertain (`board_uncertain`, final progress unscorable, peak progress still valid);
3. opens the gripper in place and returns the arm to joint home, like every other stop.

The check runs on the measured stroke, so it also fires during the close itself if the stroke passes 8 mm on
the way down, which only happens when nothing stops the fingers.

## 4. The other automatic stops

Every stop below ends the trial, opens the gripper where the arm is, and homes. None of them changes a policy
command; a command the client will not execute is logged as `rejected` with its reason and the run stops.

| stop | rule | status |
|---|---|---|
| goal reached | the tracked board equals the goal after the last release dwell (`--stop-when-solved`) | `task_solved` |
| workspace | a segment leaves the XYZ box in `cosmos_workspace.json` or exceeds the motion limits | `rejected_command` |
| carry height | a sideways segment with the jaw closed below `--carry-min-z-m` (170 mm) | `rejected_command` |
| tracking | measured position more than `--max-tracking-error-m` (8 mm) from the dispatched segment | `failed` |
| orientation | tool tilt beyond `--max-orientation-error-deg` (3 degrees) while moving | `failed` |
| stale camera | a frame older than 50 ms in live mode | `failed` |
| move budget (arm protocol) | 2 x goal distance ring moves without reaching the goal | `budget_spent` |
| illegal placement (arm protocol) | a ring released on a smaller one | `illegal_move` |
| stall (arm protocol) | `--stall-s` (60 s) without a completed ring move | `stalled` |
| duration (default rules) | `--duration-s` elapsed; a held ring is first placed within `--finish-grace-s` | `duration_reached` |
| operator | Ctrl-C | `operator_stop` |

The arm protocol's rules apply to live runs against a play-trained server or with `--distance` below 15
(`--trial-rules` overrides); otherwise the duration rule applies. Between `--repeats` the client plans the
shortest sequence of moves from the tracked board back to the start tower and performs them itself as scripted
pick-and-place (`board_reset.py`); an unknown or illegal board, a missed grasp or a leg that does not arrive
hands the reset to the operator.

## 5. Known limits

- A ring that leaves the jaw without the stroke dropping below 8 mm (for example if the fingers close on the
  ring's edge) is not detected; the tracker then believes a move that did not happen. The jaw stroke logged
  with every `grasp` event and the camera bag are the evidence to check in doubtful trials.
- Peg assignment uses the commanded y only. A release between two columns is assigned to the nearer threshold,
  not flagged.
- The board after an illegal stacking is physically possible but not a Hanoi state, so distances from it are
  undefined; the trial keeps its peak progress and, under the arm protocol, ends there.
