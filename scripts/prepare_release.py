#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "edsnlp" / "_version.py"
RELEASE_MESSAGE_START = "<!-- release-message:start -->"
RELEASE_MESSAGE_END = "<!-- release-message:end -->"
VERSION_RE = re.compile(r'^__version__\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
CHANGELOG_VERSION_RE = re.compile(
    r"^## v(\d+\.\d+\.\d+) \(\d{4}-\d{2}-\d{2}\)$", re.MULTILINE
)


class ReleaseError(RuntimeError):
    pass


def placeholder_release_notes() -> str:
    return "## Pull Requests\n\n_To be completed before merge._"


@dataclass
class PreparedRelease:
    version: str
    tag: str
    branch: str
    previous_tag: str | None
    changelog_path: Path
    changelog_body: str
    touched_files: list[Path]


def run(
    *args: str,
    cwd: Path = ROOT,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=capture_output,
    )
    if check and proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        details = stderr or stdout or f"{args[0]} exited with code {proc.returncode}"
        raise ReleaseError(details)
    return proc


def normalize_version(version: str) -> str:
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ReleaseError(
            f"Invalid version '{version}'. Expected a semantic version like 0.21.0."
        )
    return version


def bump_version(current: str, part: str) -> str:
    major, minor, patch = [int(value) for value in current.split(".")]
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ReleaseError(f"Unsupported bump part: {part}")


def tracked_changelog_path() -> Path:
    tracked_files = run("git", "ls-files").stdout.splitlines()
    for path in tracked_files:
        if path.lower() == "changelog.md":
            return ROOT / path
    raise ReleaseError("Could not find a tracked changelog.md file.")


def read_current_version() -> str:
    match = VERSION_RE.search(VERSION_FILE.read_text())
    if match is None:
        raise ReleaseError(f"Could not find __version__ in {VERSION_FILE}.")
    return match.group(1)


def first_released_version(changelog_text: str) -> str | None:
    match = CHANGELOG_VERSION_RE.search(changelog_text)
    return None if match is None else match.group(1)


def rename_unreleased(changelog_text: str, version: str, release_date: str) -> str:
    updated = re.sub(
        r"^## Unreleased$",
        f"## v{version} ({release_date})",
        changelog_text,
        count=1,
        flags=re.MULTILINE,
    )
    if updated == changelog_text:
        raise ReleaseError("Could not find '## Unreleased' in changelog.")
    return updated


def extract_release_changelog(
    changelog_text: str, version: str, release_date: str
) -> str:
    pattern = re.compile(
        rf"^## v{re.escape(version)} \({re.escape(release_date)}\)\n(.*?)(?=^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(changelog_text)
    if match is None:
        raise ReleaseError(
            f"Could not extract changelog section for v{version} ({release_date})."
        )
    return match.group(1).strip()


def normalize_generated_notes(notes: str) -> str:
    normalized = re.sub(
        r"^## What's Changed$",
        "## Pull Requests",
        notes.strip(),
        count=1,
        flags=re.MULTILINE,
    )
    return normalized.strip()


def render_release_message(changelog_body: str, generated_notes: str) -> str:
    parts = [
        "<!-- Edit the release message between the markers before merging this PR. -->",
        RELEASE_MESSAGE_START,
        "## Changelog",
        "",
        changelog_body.strip(),
    ]
    if generated_notes.strip():
        parts.extend(["", generated_notes.strip()])
    parts.append(RELEASE_MESSAGE_END)
    return "\n".join(parts).strip() + "\n"


def working_tree_is_dirty() -> bool:
    tracked_changes = run(
        "git",
        "status",
        "--porcelain",
        "--untracked-files=no",
    ).stdout.strip()
    return bool(tracked_changes)


def ensure_release_branch(branch: str) -> None:
    current_branch = run("git", "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if current_branch == branch:
        return
    existing_branches = set(run("git", "branch", "--list", branch).stdout.split())
    if branch in existing_branches:
        run("git", "switch", branch, capture_output=False)
    else:
        run("git", "switch", "-c", branch, capture_output=False)


def files_with_version(version: str) -> list[Path]:
    proc = run("git", "grep", "-l", "-F", version, "--", ".", check=False)
    if proc.returncode not in (0, 1):
        raise ReleaseError(proc.stderr.strip() or proc.stdout.strip())

    files = []
    for line in proc.stdout.splitlines():
        path = ROOT / line
        if path.name.lower() == "changelog.md":
            continue
        files.append(path)
    return sorted(set(files))


def replace_version_mentions(old_version: str, new_version: str) -> list[Path]:
    touched_files: list[Path] = []
    for path in files_with_version(old_version):
        content = path.read_text()
        updated = content.replace(old_version, new_version)
        if updated != content:
            path.write_text(updated)
            touched_files.append(path)
    return touched_files


def write_release_message_file(body: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".md",
        delete=False,
    )
    with handle:
        handle.write(body)
    return Path(handle.name)


def git_remote_repo() -> str:
    origin = run("git", "remote", "get-url", "origin").stdout.strip()
    match = re.search(r"github\.com[:/]([^/]+/[^/.]+?)(?:\.git)?$", origin)
    if match is None:
        raise ReleaseError(
            f"Could not extract GitHub repository from origin URL: {origin}"
        )
    return match.group(1)


def generate_release_notes(
    repo: str,
    tag: str,
    previous_tag: str | None,
    target_commitish: str,
) -> str:
    if shutil.which("gh") is None:
        return placeholder_release_notes()

    command = [
        "gh",
        "api",
        "--method",
        "POST",
        f"repos/{repo}/releases/generate-notes",
        "-f",
        f"tag_name={tag}",
        "-f",
        f"target_commitish={target_commitish}",
    ]
    if previous_tag is not None:
        command.extend(["-f", f"previous_tag_name={previous_tag}"])

    proc = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return placeholder_release_notes()

    body = json.loads(proc.stdout).get("body", "").strip()
    if not body:
        return placeholder_release_notes()
    return normalize_generated_notes(body)


def commit_changes(paths: Iterable[Path], version: str) -> None:
    relative_paths = [str(path.relative_to(ROOT)) for path in paths]
    if not relative_paths:
        raise ReleaseError("No files were updated.")
    run("git", "add", "--", *relative_paths, capture_output=False)
    diff = run("git", "diff", "--cached", "--name-only").stdout.strip()
    if not diff:
        raise ReleaseError("No staged changes were found after updating files.")
    run(
        "git", "commit", "-m", f"chore: bump version to {version}", capture_output=False
    )


def create_or_update_pr(
    branch: str,
    base: str,
    title: str,
    body: str,
) -> None:
    if shutil.which("gh") is None:
        raise ReleaseError("gh CLI is required to create or update the PR.")

    body_file = write_release_message_file(body)
    try:
        existing = subprocess.run(
            ["gh", "pr", "view", branch, "--json", "number"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        if existing.returncode == 0:
            pr_number = json.loads(existing.stdout)["number"]
            run(
                "gh",
                "pr",
                "edit",
                str(pr_number),
                "--title",
                title,
                "--body-file",
                str(body_file),
                capture_output=False,
            )
            return

        run(
            "gh",
            "pr",
            "create",
            "--base",
            base,
            "--head",
            branch,
            "--title",
            title,
            "--body-file",
            str(body_file),
            capture_output=False,
        )
    finally:
        body_file.unlink(missing_ok=True)


def prepare_release(
    version: str,
    branch: str,
    release_date: str,
) -> PreparedRelease:
    current_version = read_current_version()
    if current_version == version:
        raise ReleaseError(f"The repository is already on version {version}.")

    changelog_path = tracked_changelog_path()
    changelog_text = changelog_path.read_text()
    previous_version = first_released_version(changelog_text)
    previous_tag = None if previous_version is None else f"v{previous_version}"

    touched_files = replace_version_mentions(current_version, version)

    updated_changelog = rename_unreleased(changelog_text, version, release_date)
    changelog_path.write_text(updated_changelog)
    touched_files.append(changelog_path)

    changelog_body = extract_release_changelog(updated_changelog, version, release_date)
    tag = f"v{version}"

    return PreparedRelease(
        version=version,
        tag=tag,
        branch=branch,
        previous_tag=previous_tag,
        changelog_path=changelog_path,
        changelog_body=changelog_body,
        touched_files=sorted(set(touched_files)),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a release branch, commit, and PR for edsnlp"
    )
    parser.add_argument(
        "version",
        nargs="?",
        help="Target version, for example 0.21.0",
    )
    parser.add_argument(
        "--bump",
        choices=["major", "minor", "patch"],
        help="Compute the target version from the current one",
    )
    parser.add_argument(
        "--base",
        default="master",
        help="Base branch for the release PR",
    )
    parser.add_argument(
        "--branch",
        help="Release branch name. Defaults to release/v<version>",
    )
    parser.add_argument(
        "--release-date",
        default=dt.date.today().isoformat(),
        help="Release date inserted into changelog headings",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Create the release commit after updating files",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the release branch to origin. Requires --commit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force push the release branch to origin. Requires --push",
    )
    parser.add_argument(
        "--pr",
        action="store_true",
        help="Create or update the release PR. Requires --push",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow tracked local changes before preparing the release",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    current_version = read_current_version()

    if args.version and args.bump:
        raise ReleaseError("Pass either a target version or --bump, not both")
    if not args.version and not args.bump:
        raise ReleaseError("Pass a target version or use --bump")

    version = (
        normalize_version(args.version)
        if args.version
        else bump_version(current_version, args.bump)
    )
    branch = args.branch or f"release/v{version}"

    if working_tree_is_dirty() and not args.allow_dirty:
        raise ReleaseError(
            "Tracked local changes detected. Commit or stash them first, or rerun "
            "with --allow-dirty"
        )
    if args.push and not args.commit:
        raise ReleaseError("Use --push together with --commit")
    if args.pr and not args.push:
        raise ReleaseError("Use --pr together with --push")

    ensure_release_branch(branch)
    prepared = prepare_release(
        version=version, branch=branch, release_date=args.release_date
    )

    if args.commit:
        commit_changes(prepared.touched_files, prepared.version)
    if args.push:
        cmd = ["git", "push", "-u", "origin", prepared.branch]
        if args.force:
            cmd.append("--force")
        run(*cmd, capture_output=False)

    repo = git_remote_repo()
    target_commitish = prepared.branch if args.push else args.base
    generated_notes = generate_release_notes(
        repo=repo,
        tag=prepared.tag,
        previous_tag=prepared.previous_tag,
        target_commitish=target_commitish,
    )
    release_message = render_release_message(prepared.changelog_body, generated_notes)

    if args.pr:
        create_or_update_pr(
            branch=prepared.branch,
            base=args.base,
            title=f"chore: bump version to {prepared.version}",
            body=release_message,
        )

    print(f"Prepared release {prepared.tag} on branch {prepared.branch}")
    print("Touched files:")
    for path in prepared.touched_files:
        print(f"- {path.relative_to(ROOT)}")
    if not args.pr:
        print("\nRelease message:")
        print(release_message)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
