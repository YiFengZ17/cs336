> 2026-11-10 存储迁移已完成。当前路径、更新和下载命令见 [存储与同步指南](docs/storage-workflow.md)；下文旧相对路径通过兼容链接继续使用，服务器信息以新指南为准。

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
