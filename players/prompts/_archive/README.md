# Superseded prompt templates

Every run record carries `provenance.prompt_template_sha256` — the first 16 hex
characters of the SHA-256 of the template file that was actually sent. That field
is only worth recording if the text it names can still be read, so a template is
never edited in place and discarded: the previous bytes land here as

    <template_name>.<sha16>.txt

and stay forever. The live template keeps its own name one directory up.

**Why this exists.** Editing `prereg_deviation.txt` on 2026-08-15 (adding an
explicit "return exactly these keys and no others" instruction) would otherwise
have made `cbdd21b9a57849d2` — stamped on 92 published records — a hash of text
that no longer existed anywhere. A provenance field pointing at nothing is worse
than no field, because it looks like provenance.

**Two guards depend on this directory**, both in `framework/tests/test_prompts.py`:

* `test_every_published_prompt_hash_resolves_to_a_prompt_file` — every distinct
  `prompt_template_sha256` in `arenas/*/runs/**` must match some template here or
  in the live directory. Deleting history turns the build red.
* `test_one_arena_and_task_set_never_mixes_prompt_templates` — within a single
  (arena, task set), every record that carries a hash must carry the *same* one.
  A player measured under a different instruction is not comparable with the
  others, and the leaderboard has no way to show that. This is what forces a
  full re-run of the affected players whenever a template changes.

Archived templates are mirrored to the public repo alongside the live ones.

**Archived because the task set changed, not because the prompt was edited.**
`publication_bias.c2ded7861e854256.txt` is the prompt all 72 published
publication-bias-v1 **v1** records were measured with (a bias verdict). Task set
v2 asks a different question (the statistics), so its players read
`publication_bias_v2.txt`; v1 is superseded and the runner refuses new v1 records.
`test_published_records_were_measured_with_the_CURRENT_template` therefore checks
each arena's current task set only.
