# Prompt strategies

The evaluation scripts use one of two prompt strategies for WordNet
representatives.

## Vanilla prompt

Every concept uses its own display name:

```text
a photo of <name>
```

## Prompt with father

For a concept at level `i`, append the representative of its father at the
next higher (coarser) level. In `WNTree.json`, concepts become coarser as their
number decreases, so the father of a level `i` concept is its representative
at level `i - 1`.

```text
a photo of <name>, which is one kind of <father>
```

Level 1 has no higher level and therefore uses the vanilla prompt without an
additional clause. For example, a level 3 concept uses its corresponding
level 2 representative as `<father>`.

## API

Prompt construction is centralized in:

```python
from wordnet_tree import build_prompt

build_prompt(name, tree, prompt_father_level=None)
```

The arguments are:

- `name`: the display name of the concept.
- `tree`: the parsed `WNTree.json` object.
- `prompt_father_level`: `None` for the vanilla prompt, or an integer level
  identifying the level of representative used as the father.

Examples:

```python
build_prompt("tiger cat", tree, None)
# "a photo of tiger cat"

build_prompt("tiger cat", tree, 2)
# "a photo of tiger cat, which is one kind of <level-2 representative>"
```
