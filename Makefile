
.ONESHELL:
	SHELL:=/bin/bash 

.PHONY: create-env install documentation test

default:
	@echo "Call a specific subcommand: create-env,install,documentation,test"

.venv:
	python3 -m venv .venv
	
create-env: .venv

install : .venv
	. .venv/bin/activate
	pip install -r requirements.txt
	pip install -r requirements-setup.txt
	python3 scripts/conjugate_verbs.py
	pip install -r requirements-dev.txt
	pip install -r requirements-docs.txt
	pip install -e .
	pre-commit install

documentation: .venv
	. .venv/bin/activate
	pip install -r requirements-docs.txt
	mkdocs serve

test: .venv
	. .venv/bin/activate
	python3 -m pytest

