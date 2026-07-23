# Getting this onto GitHub

The repo is already initialised with seven atomic commits. What's left is
creating the remote and pushing.

---

## 1. Fix the commit authorship first

I committed under a placeholder identity. Set yours before pushing, or every
commit will be attributed to `you@example.com` and won't link to your profile.

```bash
cd gamestate

git config user.name  "Your Name"
git config user.email "your-github-email@example.com"
```

Use the email attached to your GitHub account, or your GitHub noreply address
(`12345678+DrDraken01@users.noreply.github.com`) if you'd rather not publish a
real one. It's on GitHub under Settings → Emails.

Then rewrite the existing commits to match:

```bash
git rebase --root --exec 'git commit --amend --no-edit --reset-author'
```

Do this *before* the first push. After, it rewrites published history.

---

## 2. Create the remote

**With the GitHub CLI** (easiest — install from cli.github.com):

```bash
gh auth login
gh repo create gamestate --public --source=. --remote=origin --push
```

**Without it:** create an empty repo on github.com — no README, no .gitignore,
no license, since we already have all three — then:

```bash
git remote add origin git@github.com:DrDraken01/gamestate.git
git branch -M main
git push -u origin main
```

---

## 3. Use SSH, not HTTPS

HTTPS means pasting a personal access token, which ends up in your shell
history and your credential helper. SSH keys don't.

```bash
ssh-keygen -t ed25519 -C "your-github-email@example.com"
cat ~/.ssh/id_ed25519.pub          # paste into GitHub → Settings → SSH keys
ssh -T git@github.com              # should greet you by username
```

---

## 5. Turn on branch protection

Settings → Branches → Add rule for `main`:

- Require a pull request before merging
- Require status checks to pass → select `quality` and `security`
- Require branches to be up to date before merging

Protecting main in a solo repo feels absurd. It isn't. It's what makes CI
*mean* something — otherwise a red build is a notification you can ignore, and
you will. It also forces the branch-per-change habit that makes `git bisect`
useful later.

---

## 6. Local setup on a fresh clone

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest
```

---

## Working rhythm

This is the "one file at a time, test before and after" discipline you
described, expressed in git:

```bash
git switch -c feat/rest-days      # branch per change
pytest                            # green BEFORE you touch anything
# ... make one focused change ...
pytest                            # green AFTER
git add -p                        # stage in hunks, review as you go
git commit -m "feat(features): add rest-day differential"
git push -u origin feat/rest-days
gh pr create --fill
```

Two notes on this.

**Branches replace file backups.** `cp features.py features_backup.py` clutters
the tree, gets committed by accident, and can't be diffed usefully. A branch is
a free, named, revertible checkpoint — and if a change turns out to have broken
something three weeks later, `git bisect` will find the exact commit in about
eight steps. That only works if commits are atomic, which is why the "never
batch" rule matters mechanically and not just aesthetically.

**`git add -p` is underused.** It walks you through each hunk and asks whether
to stage it. It's the cheapest code review that exists, and it catches stray
debug prints constantly.

---

## Before the site goes public

- **Never commit `.env`.** It's gitignored; keep it that way. Secrets go in
  your host's environment variable settings.
- **`.secrets.baseline` IS committed on purpose** — it's the allowlist of
  reviewed findings, not a secret itself.
- **If you ever do leak a key: rotate it, don't just delete the commit.**
  History rewriting doesn't un-leak anything. Clones, forks, and GitHub's own
  cache already have it. The key is compromised from the moment it's pushed.
- **Data stays out of git.** `data/` is ignored. Parquet caches are
  regenerable, and repos that swallow data files become unclonable fast.
