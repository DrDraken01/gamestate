# START HERE — GameState on a fresh Mac

You already have Homebrew. Everything below is the rest, in order.

---

## 1. Put the project somewhere permanent

Download `gamestate.zip` from the conversation, then:

```bash
ls ~/Downloads | grep -i gamestate
```

**If it shows `gamestate.zip`:**

```bash
cd ~/Downloads
unzip gamestate.zip
mkdir -p ~/projects
mv gamestate ~/projects/
cd ~/projects/gamestate
```

**If it shows `gamestate` with no `.zip`** — Safari auto-expands archives, so
it's already unzipped. Skip the `unzip` line.

Verify the download survived intact:

```bash
ls -a          # expect .git .github .gitignore .pre-commit-config.yaml src tests docs
git log --oneline
```

Seven commits, most recent first. If `.git` is missing, the history didn't
survive — say so and we'll re-init.

---

## 2. Install the toolchain

```bash
brew install python@3.12 git gh
```

`gh` is GitHub's CLI; it makes repo creation a one-liner later.

```bash
python3.12 --version     # 3.12.x
git --version
gh --version
```

Note that plain `python3` will probably still report Apple's 3.9.6. That's
fine and expected — leave Apple's alone, it belongs to the OS.

---

## 3. Create the virtual environment

```bash
cd ~/projects/gamestate
python3.12 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`.

```bash
python --version         # 3.12.x — plain `python` works now
which python             # .../gamestate/.venv/bin/python
```

The version you name at *creation* is baked in permanently. That's the only
moment it matters.

---

## 4. Install and verify

```bash
pip install --upgrade pip
pip install -e ".[dev]"
pre-commit install
pytest
```

Expected: `10 passed`.

Then run the real thing — it downloads ~2 MB from nflverse and backtests
4,338 games:

```bash
python scripts/run_slice.py
```

You should see the base rate / model / market comparison table.

---

## 5. Push to GitHub

```bash
gh auth login            # choose SSH when asked
gh repo create gamestate --public --source=. --remote=origin --push
```

Then: `docs/github-setup.md` for branch protection and the branching workflow.

---

## Daily rhythm

Every new terminal window needs the venv activated. It is not persistent, and
forgetting is the single most common cause of "it worked yesterday":

```bash
cd ~/projects/gamestate
source .venv/bin/activate
```

Working on a change:

```bash
git switch -c feat/whatever
pytest                    # green BEFORE
# ... one focused change ...
pytest                    # green AFTER
git add -p                # stage in hunks, review as you go
git commit -m "feat(scope): what changed"
```

---

## If something breaks

**`command not found: python`** — venv isn't active. `source .venv/bin/activate`.

**`command not found: brew`** in a new window — the PATH line didn't persist:
```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
source ~/.zprofile
```

**`xcrun: error: invalid active developer path`** — `xcode-select --install`

**pyarrow tries to compile from source** — stale pip. `pip install --upgrade pip` first.

**`pre-commit run --all-files` fails on first run** — expected. It auto-fixes
formatting and reports failure so you review the changes. Run it again.

**`run_slice.py` network error** — nflverse is a GitHub release; check
connectivity, then retry. Data caches to `data/raw/` so it only downloads once.
