# CS336

Stanford CS336 (Spring 2026) assignments, linked as git submodules.

| Assignment | Repository |
| --- | --- |
| 1 — Basics | [assignment1-basics](https://github.com/YiFengZ17/assignment1-basics) |
| 2 — Systems | [assignment2-systems](https://github.com/YiFengZ17/assignment2-systems) |
| 3 — Scaling | [assignment3-scaling](https://github.com/YiFengZ17/assignment3-scaling) |
| 4 — Data | [assignment4-data](https://github.com/YiFengZ17/assignment4-data) |
| 5 — Alignment | [assignment5-alignment](https://github.com/YiFengZ17/assignment5-alignment) |

## Clone

```bash
git clone --recurse-submodules https://github.com/YiFengZ17/cs336.git
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Updating an assignment

Each submodule has `origin` pointing at the fork and `upstream` pointing at
`stanford-cs336`. Work and push inside the submodule, then commit the updated
pointer here:

```bash
cd assignment1-basics
git add -A && git commit -m "..." && git push
cd ..
git add assignment1-basics && git commit -m "Update assignment1 pointer"
git push
```
