# edges DSL

`edges` declares what data an execution unit receives and where each piece comes from. It maps a variable-set name to a single string:

```python
edges = {
    'X': '{age, income} + scale:(*) + ohe:(*)',
    'y': '{target}',
    'sample_weight': '{w}',
}
```

Keys are yours to name — `'X'`, `'y'`, `'sample_weight'` are the ones processors and adapters look for. Every value is **always a plain string**, in the definition, through group inheritance, in storage, and in comparison.

## Columns appear only at run time

Defining a node validates the string's *structure* — that it parses, and that any node it names exists. Nothing looks at a schema and no column list is produced.

The expansion into real column names happens once, inside the processor, against the actual data. So a node may legitimately reference columns that will not exist until an upstream processor creates them, and the same definition can be reused on data whose column set differs.

## Grammar

```
expr        := term (op term)*          op is '+' / '-' / '&'
term        := ('*' | set_literal | pattern) ['@' NAME]
             | slice | namespace | '(' expr ')'
set_literal := '{' [NAME (',' NAME)*] '}'
namespace   := NAME ':' '(' expr ')'
slice       := [INT] ':' [INT]
pattern     := REGEX
```

All operators share one precedence and associate left to right.

| Form | Selects |
|---|---|
| `*` | every column |
| `{a, b}` | exactly those columns; unknown names raise |
| `A.*` | columns matching the regex, via `re.match` |
| `1:` / `:-1` / `0:3` | by position, Python slice semantics |
| `node:(expr)` | evaluate `expr` against that node's output |
| `(expr)` | grouping |

| Operator | |
|---|---|
| `a + b` | union, concatenated column-wise |
| `a - b` | difference |
| `a & b` | intersection |

!!! warning "Put spaces around `+`, `-` and `&`"
    Regex already uses those characters. An operator is recognised only as a standalone, whitespace-bounded token — `A.*-B` is one pattern, `A.* - B.*` is a difference.

## Namespaces: where columns come from

A bare term refers to the **DataSource**. A `name:(...)` block refers to that node's output.

```python
'{age, income} + scale:(*)'          # two raw columns, plus everything scale produced
'scale:(* - {age})'                  # scale's output minus one column
'ohe:(A.*) + tgt:(*)'                # regex inside one node, all of another
```

Only `+` may join top-level segments. `-` and `&` have to sit inside a namespace or a parenthesised group, because subtracting one node's output from another's is rarely what anyone means.

!!! note "Name DataSource columns explicitly"
    For DataSource-origin terms, prefer an explicit `{a, b}` set literal over `*` or a pattern. The raw frame usually carries columns the pipeline should not see — ids, targets, sample weights — and `*@numeric` at that level will happily pull them in.

## `@selector` suffixes

A selector, attached directly after `*`, a set literal or a pattern with no space, filters the columns that expression already matched.

```python
'*@numeric'                  # every numeric column
'{a, b, c}@int'              # of those three, the integer ones
'ohe:(*@ohe_drop_first)'     # processor-aware: drop each feature's first level
```

Built-in dtype selectors need no processor: `@numeric`, `@categorical`, `@binary` (bool dtype only), `@float`, `@int`, `@string`.

Others are registered against a processor type and are valid only there — `@ohe_drop_first` for `OneHotEncoder`, `@subset_poly` for `PolynomialFeatures`. Using one outside its processor raises `ValueError`.

Register your own with the `col_selector` decorator; every selector has the signature `(data, processor=None) -> mask` and receives `data` already narrowed to the candidate columns. Selectors take no arguments.

!!! warning "A selector needs something to filter"
    `@` attaches to the expression in front of it, so it cannot open a term:

    ```python
    'ohe:(@ohe_drop_first)'    # invalid — nothing before the @
    'ohe:(*@ohe_drop_first)'   # correct
    ```

## Inheritance

When a node belongs to a group, an edge value starting with `+` or `-` **extends** the resolved parent value; anything else replaces it, and an absent key inherits the parent's as-is.

```python
p.set_grp('model', edges={'X': '{age, income}', 'y': '{target}'})
p.set_node('m1', grp='model')                            # X = '{age, income}'
p.set_node('m2', grp='model', edges={'X': '+ scale:(*)'})  # X = '{age, income} + scale:(*)'
p.set_node('m3', grp='model', edges={'X': 'ohe:(*)'})      # X = 'ohe:(*)'
```

## X-less units

A unit whose edges have `'y'` but no `'X'` — a `LabelEncoder`, say — uses `y` as its primary input; the output becomes the new `y` columns.

## At inference time

`Inferencer` resolves **only `'X'` edges**. `'y'` and `'sample_weight'` are training-only and get dropped, which is what lets the same definitions run on data that has no target column.

## Working with the strings

```python
from mllabs._edge_dsl import parse, unparse, referenced_nodes, iter_segments

referenced_nodes('{age} + scale:(*)')   # {None, 'scale'} — None is the DataSource
list(iter_segments('{age} + scale:(*)'))
```

`Pipeline.get_node_names()` and `desc_node()` are the usual way to see what a node actually reads.
