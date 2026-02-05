# AMD Skills

Agent skills collection for AMD hardware development workflows.

## Available Skills

- **rocprofv3-profiler**: Profile AMD GPU kernels and analyze performance bottlenecks

## Usage as Git Submodule

### Add to Your Project
```bash
cd /path/to/your/project
git submodule add https://github.com/Arist12/AMD-Skills.git .claude/skills
git commit -m "feat: add AMD Skills submodule"
```

This places the contents of AMD-Skills directly into `.claude/skills/`.

### Clone a Project with Submodules
```bash
git clone --recurse-submodules <your-project-url>
```

Or if already cloned without `--recurse-submodules`:
```bash
git submodule update --init
```

This checks out the submodule to the exact commit recorded by the parent project (detached HEAD state).

---

## Two Workflows

### Workflow A: Co-evolve Skills (Active Development)

Use this when you want to modify skills and push changes back to the skills repo.

**Initial setup** — switch to a branch (avoid detached HEAD):
```bash
cd .claude/skills
git checkout main
```

**Make changes:**
```bash
cd .claude/skills
# ... edit skills ...
git add .
git commit -m "Improve skill"
git push origin main
```

**Then update the parent project's pointer:**
```bash
cd ../..
git add .claude/skills
git commit -m "Update skills submodule"
```


**Pull latest changes from skills repo:**
```bash
cd .claude/skills
git pull origin main
cd ../..
git add .claude/skills
git commit -m "Update skills to latest"
```

### Workflow B: Lock to Specific Version (Consumer Mode)

Use this when you just want to use skills at a known-good version without modifying them.

**Sync to the version recorded in parent project:**
```bash
git submodule update --init
```

**Update to latest version from remote:**
```bash
git submodule update --remote
git add .claude/skills
git commit -m "Update skills to latest"
```

Both commands result in detached HEAD state, which is expected — you're checking out a specific commit, not working on a branch.

---

## Quick Reference

| Goal | Command |
|------|---------|
| Add submodule | `git submodule add <url> .claude/skills` |
| Clone with submodules | `git clone --recurse-submodules <url>` |
| Initialize after clone | `git submodule update --init` |
| Switch to branch for dev | `cd .claude/skills && git checkout main` |
| Pull latest (dev mode) | `cd .claude/skills && git pull` |
| Update to latest (consumer mode) | `git submodule update --remote` |

## Common Pitfalls

1. **Detached HEAD when you want to develop**: Run `git checkout main` inside the submodule before making changes.

2. **Forgot to commit in parent after submodule changes**: Your teammates won't see the update until you commit the new pointer in the parent repo.

3. **Teammates get empty submodule folder**: They need to run `git submodule update --init` after pulling.
