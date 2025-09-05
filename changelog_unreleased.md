### Added
- New `edsnlp.external_information_qualifier` qualifies spans in a document based on external information and a defined distance to these contextual/external elements as in Distant Supervision.
- New `eds.contextual_qualifier` pipeline component to qualify spans based on contextual information.
- Add the fixture `edsnlp_blank_nlp` for the test.

### Fixed
- Correct the contributing documentation. Delete `$ pre-commit run --all-files`recommendation.
- Fix the the `Obj Class` in the doc template `class.html`.
- Fix the `get_pipe_meta` function.
