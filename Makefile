
.ONESHELL:
	SHELL:=/bin/bash

.PHONY: create-env install documentation test

default:
	@echo "Call a specific subcommand: create-env,install,documentation,test"

.venv:
	python -m venv .venv

create-env: .venv

install : .venv
	. .venv/bin/activate
	pip install -r '.[dev,setup]'.txt
	python scripts/conjugate_verbs.py
	pip install -e .
	pre-commit install

documentation: .venv
	. .venv/bin/activate
	pip install -e '.[dev]'
	mkdocs serve

test: .venv
	. .venv/bin/activate
	python -m pytest
