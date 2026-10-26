# Working notes

## What this repository is

One job: read Movebank GPS and acceleration exports (`.csv` or `.csv.gz`),
classify each burst with a small trained model, and write one output CSV. It is
run by a third party on their own machine, on many files at a time. No
training, no experiments, no cluster, no database.

The model input is a burst of 20 acceleration samples at 20 Hz (50 ms apart)
plus the ground speed of the matching GPS fix. A run of 60 clean samples is 3
bursts, each with its own predicted class.

## Rules

- Answer from this repository only. The other repos in this VS Code workspace
  (`bird-behavior`, `hedge-seg`, ...) are not context unless I say so.
  This repo started from `bird-behavior` but is now independent: read a file
  there only when I point at one, and do not point to its files from this
  repo's code or docs. Skills too: use only
  this project's, in `~/dev/gulliver-behavior-classifier/.claude/skills`,
  because the other repos have same-named ones.
- **A third party runs this code.** Nothing in the package may assume my
  machine: no `/home/fatemeh` paths, no database URL, no credentials, no
  environment variables I happen to have set. Every input, output and model
  path is a command-line argument with a sensible default inside the repo.
- The command-line entry point takes real arguments (`argparse` from the
  standard library), not a config dict edited in place: a third party cannot
  be asked to edit source to run a file.
- Keep the dependency list small. Every extra package is one more thing the
  third party has to install. Ask before adding one; once approved, add it to
  `pyproject.toml` with a short comment saying what it is for.
- Change only what was asked. Leave unrelated code, formatting, files and
  wording alone, even when something nearby looks improvable — mention it
  instead.
- Before returning a result, check it. Run the tests, and run the code on the
  real example files (see Workspace). Verify every number and claim against the
  source file or a computation, rather than assuming it followed from the
  previous step.
- Explain short, in chat and in the docs. Plain words, short sentences, no
  clever phrasing. Spell out an acronym the first time (IMU, GPS). If a
  sentence needs re-reading, rewrite it.
- Tests live in `tests/`, one behaviour per test, built from small DataFrames
  written to `tmp_path`. Test the filters — a burst is skipped for a reason,
  and the reason is counted. Do not add a test that needs a file from my
  Downloads folder.
- When reporting a path, give a full clickable path so it is easy to check.
  Name the exact function or class too, not "the loader".
- Say which files were inspected before answering a question about the repo.
- Always call python by its full path:
  `/home/fatemeh/miniconda3/envs/gulliver/bin/python`. Not bare `python`, not
  `conda run -n gulliver`. VS Code puts one env's `bin` at the front of `PATH`
  for every tool call, and conda does not know it is there, so `conda run` and
  `conda activate` can silently run the wrong env. A full path cannot be
  shadowed. Use only this env, never another one such as `bird`, even when
  this one lacks a package; ask instead.
- After each change, format, check and test. The tools are not on `PATH`, so
  call them from the env too:

  ```bash
  B=/home/fatemeh/miniconda3/envs/gulliver/bin
  for i in gulliver tests; do echo $i; $B/black $i -l 88; $B/isort $i --profile black; $B/pyflakes $i; done
  $B/python -m pytest
  ```
- Markdown is wrapped at 80 columns by hand. `mdformat` is not installed in
  this env; do not reach for it.

## Commits

- Never run `git commit`. Prepare the change and the message; I commit.
- Commit messages: never mention Claude or Claude Code (no attribution, no
  Co-Authored-By line, no tool names).
- A commit message must cover everything still uncommitted (`git status`), not
  only the latest change. Check what is pending before writing it.
- Always end a change with a ready-to-paste commit message, without being
  asked, in a copyable block. **Keep it short**: a ~50-character imperative
  subject and a body of a few lines wrapped at 72. Say what changed and why.
  Say so if the change is not worth committing.
- Give it as the full command, `git commit -m "..."`, not as bare text I have
  to wrap myself. **So the message may not contain any of these four
  characters:**

  | character | why it breaks |
  |---|---|
  | `"` | closes the quote early |
  | `'` | breaks the shell if I quote with it instead |
  | `` ` `` | runs as a command inside double quotes |
  | `$` | expands as a variable inside double quotes |

  Write `model=BirdModelSmallDilated`, not `` `model="..."` ``. Say: the 0.05
  threshold, without quotes. Blank lines between paragraphs are fine inside the
  quotes, so a multi-paragraph message still works with one `-m`.

## Workspace

- Example third-party data, the files to test a change against:

  | file | what |
  |---|---|
  | `/home/fatemeh/Downloads/gulliver/data_unzip/HG_TEXEL-gps-2019-subset.csv` | small, 3 birds, January and March 2019 |
  | `/home/fatemeh/Downloads/gulliver/data_unzip/HG_TEXEL-acceleration-2019-subset.csv` | small, matches the above |
  | `/home/fatemeh/Downloads/gulliver/data/HG_TEXEL-gps-2019.csv.gz` | full year, gzipped |
  | `/home/fatemeh/Downloads/gulliver/data/HG_TEXEL-acceleration-2019.csv.gz` | full year, gzipped, 44 MB |

- The trained model is in this repo, `gulliver/models/197_best.pth`, because it
  is small (80 KB) and the third party needs it. It was trained in
  `bird-behavior` (experiment 197) and is not retrained here.
- Scratch/temporary files: use `/home/fatemeh/Downloads/bird/claude`, not `/tmp`
  (which is wiped on shutdown). Clean up whatever is not worth keeping.
- Settings/config for this project live in
  `~/dev/gulliver-behavior-classifier/.claude` (for example `settings.json`),
  never another repo's `.claude` in this VS Code workspace.

## Documentation

There are two documents, both public and written for the third party, not for
me:

- `README.md`: what to install, how to run it on one file and on many, what
  each output column means, and what makes a burst be skipped.
- `METHODS.md`: the method. The classifier runs on the data listed in
  [inbo/bird-tracking](https://github.com/inbo/bird-tracking); the data rules
  and the model come first. Training, its labeled data and its results follow,
  marked as background for reference, not something the user does. Data rules
  are bullets; model and training are prose, as in the methods section of a
  paper. No code names.

Keep both short and keep them true — prefer a table or a bullet to a
paragraph, and when adding, look for something to cut.

After a change: update the module's top docstring, `README.md` and
`METHODS.md` in the same change. The burst rules are in both documents, so
change both. A number quoted in a document goes stale silently, because
nothing breaks when it is wrong.
