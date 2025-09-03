First, when starting to develop, install the project with

```bash
pip install -e ".[dev]" "mkdocs-eds@git+https://github.com/percevalw/mkdocs-eds.git@main#egg=mkdocs-eds ; python_version>='3.9'"
pre-commit install
```

Then, when fixing an issue, add a new test to reproduce it. If the issue concerns an existing
component, add the test to the corresponding test file, or create a new test file (only when needed, this should not be the most common scenario).

Create a new branch (or checkout the auto-created branch for the issue).

Then update the codebase to fix the issue, and run the new test to check that everything is working as expected.

Update the changelog.md file with a concise explanation of the change/fix/new feature.

Before commiting, stash, checkout master and pull to ensure you have the latest version of master, then checkout the branch you were working on and rebase it on top of master.
If the rebase has changed something to the codebase, rerun the edited tests to ensure everything is still working as expected.

Finally, run git log to look at the commit messages and get an idea of what the commit messages should look like (concise, neutral, conventional commits messages).

```bash
git log --no-pager
```

Finally commit the changes.

!!! note

    Whenever you run a command, ensure that you do it without making it prompt the user for input (ie, use --no-edit in git rebase, --no-pager, --yes, etc. when possible).
