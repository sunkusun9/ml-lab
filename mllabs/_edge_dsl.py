"""DSL for specifying edges['X'|'y'|...] as a single string.

Grammar (all operators same precedence, left-associative)::

    expr        := term (op term)*            -- op is '+'/'-'/'&', a standalone
                                                   word (whitespace on both sides)
    term        := ('*' | set_literal | pattern) ['@' NAME ['(' ')']]
                    | slice | namespace | '(' expr ')'
    slice       := [INT] ':' [INT]             -- Python slice semantics, e.g.
                                                   '-1:' == slice(-1, None), ':-1'
                                                   == slice(None, -1), '1:2' == slice(1, 2)
    set_literal := '{' [NAME (',' NAME)*] '}'
    namespace   := NAME ':' '(' expr ')'
    pattern     := REGEX

A ``pattern`` is native regex text (``A.*|B.*``, ``Num[a-z]+``, ``.*_bin$``,
...), matched with ``re.match``. Because regex already uses ``+ - ( ) { }``
for its own syntax, a DSL *operator* is only recognized when it is an
isolated, whitespace-bounded token — regex quantifiers/groups/char-classes
glued directly onto surrounding pattern text (no space) are always read as
plain characters. ``(...)``/``{...}`` occurring *inside* a pattern are
tracked for balance so a regex group or ``{n,m}`` quantifier doesn't get
mistaken for DSL grouping/set-literal syntax — but a term that must start
with a literal ``*``, ``{`` or a *namespace-looking* prefix (``name:(``)
cannot be expressed as a bare pattern (rare for column-name matching; not
supported). Always put a space around ``+``/``-``/``&`` when using them as
DSL operators.

``NAME`` after ``@`` is a selector registered via ``col.col_selector`` — it
can follow ``*``, a ``set_literal``, or a ``pattern`` (whichever variable-set
expression precedes it, directly adjacent, no whitespace), receives ``data``
already narrowed to that expression's matched candidate columns, and is
applied with ``(data, processor=None)``; selectors take no arguments. Builtin
dtype selectors (``@numeric``/``@categorical``/``@binary``/``@float``/``@int``/
``@string``) need no processor — e.g. ``*@numeric`` or ``{a, b}@int``.

A bare/unnamespaced term or sub-expression refers to the DataSource; a
``name:(...)`` block refers to that stage node's output columns. Only ``+``
may join top-level segments — ``-``/``&`` must be nested inside a namespace
or parenthesized group.

``edges[key]`` is stored and passed around as this DSL string everywhere in
``Pipeline`` (definition, inheritance, serialization, comparison) — it is
never expanded into an actual column list except by the Processor itself, at
process time, against real data (:func:`eval_expr`, called from
``_flow.py.get_data``). ``set_grp``/``set_node`` only validate *structure*
(syntax + namespace references) via :func:`validate_edges` — never columns.
"""
import re

from .col import resolve_selector


class Star:
    def __init__(self, selector=None):
        self.selector = selector  # raw '@name' / '@name()' string, or None

    def __repr__(self):
        return f'Star({self.selector!r})'

    def __eq__(self, other):
        return isinstance(other, Star) and self.selector == other.selector


class SetLiteral:
    def __init__(self, names, selector=None):
        self.names = names
        self.selector = selector  # raw '@name' / '@name()' string, or None

    def __repr__(self):
        return f'SetLiteral({self.names!r}, {self.selector!r})'

    def __eq__(self, other):
        return isinstance(other, SetLiteral) and self.names == other.names and self.selector == other.selector


class Pattern:
    def __init__(self, regex, selector=None):
        self.regex = regex
        self.selector = selector  # raw '@name' / '@name(args)' string, or None

    def __repr__(self):
        return f'Pattern({self.regex!r}, {self.selector!r})'

    def __eq__(self, other):
        return isinstance(other, Pattern) and self.regex == other.regex and self.selector == other.selector


class Namespace:
    def __init__(self, name, expr):
        self.name = name
        self.expr = expr

    def __repr__(self):
        return f'Namespace({self.name!r}, {self.expr!r})'

    def __eq__(self, other):
        return isinstance(other, Namespace) and self.name == other.name and self.expr == other.expr


class BinOp:
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):
        return f'BinOp({self.op!r}, {self.left!r}, {self.right!r})'

    def __eq__(self, other):
        return (
            isinstance(other, BinOp) and self.op == other.op
            and self.left == other.left and self.right == other.right
        )


_NAME_COLON_RE = re.compile(r'[A-Za-z_]\w*(?=:\()')
_SLICE_RE = re.compile(r'(-?\d+)?:(-?\d+)?')


def _scan_selector_ref(text, start):
    """From text[start] == '@', consume '@name' or '@name()' — selectors take
    no arguments (see ``col.col_selector``)."""
    i = start + 1
    n = len(text)
    while i < n and (text[i].isalnum() or text[i] == '_'):
        i += 1
    if i == start + 1:
        raise ValueError(f"Invalid selector reference at position {start}: {text[start:start + 20]!r}")
    if i < n and text[i] == '(':
        if i + 1 < n and text[i + 1] == ')':
            i += 2
        else:
            raise ValueError(f"Selector arguments are not supported: {text[start:i + 20]!r}")
    return text[start:i], i


class _Parser:
    """Direct string+index recursive-descent parser (no separate tokenizer) —
    disambiguating DSL operators from regex-internal characters requires
    scanning position-sensitively (see module docstring)."""

    def __init__(self, text):
        self.text = text
        self.i = 0
        self.n = len(text)

    def _skip_ws(self):
        while self.i < self.n and self.text[self.i].isspace():
            self.i += 1

    def _peek_char(self):
        return self.text[self.i] if self.i < self.n else ''

    def _expect(self, ch):
        if self._peek_char() != ch:
            raise ValueError(f"Expected {ch!r} in {self.text!r} at position {self.i}")
        self.i += 1

    def parse_expr(self):
        left = self.parse_term()
        while True:
            self._skip_ws()
            op = self._try_consume_op()
            if op is None:
                break
            self._skip_ws()
            right = self.parse_term()
            left = BinOp(op, left, right)
        return left

    def _try_consume_op(self):
        # A standalone '+'/'-'/'&' (whitespace already skipped before it, and
        # whitespace/EOF right after) is an operator; otherwise it belongs to
        # a pattern (e.g. a regex quantifier) and is left for parse_pattern.
        if self._peek_char() in ('+', '-', '&'):
            nxt = self.text[self.i + 1] if self.i + 1 < self.n else ''
            if nxt == '' or nxt.isspace():
                op = self.text[self.i]
                self.i += 1
                return op
        return None

    def parse_term(self):
        self._skip_ws()
        if self.i >= self.n:
            raise ValueError(f"Unexpected end of input in {self.text!r}")
        c = self._peek_char()
        if c == '*':
            self.i += 1
            return Star(self._maybe_scan_selector())
        if c == '{':
            node = self.parse_set_literal()
            node.selector = self._maybe_scan_selector()
            return node
        if c == '(':
            self.i += 1
            expr = self.parse_expr()
            self._skip_ws()
            self._expect(')')
            return expr
        m = _NAME_COLON_RE.match(self.text, self.i)
        if m:
            name = m.group(0)
            self.i = m.end()
            self._expect(':')
            self._expect('(')
            expr = self.parse_expr()
            self._skip_ws()
            self._expect(')')
            return Namespace(name, expr)
        m = _SLICE_RE.match(self.text, self.i)
        if m and m.group(0):
            start, stop = m.group(1), m.group(2)
            self.i = m.end()
            return slice(int(start) if start else None, int(stop) if stop else None)
        return self.parse_pattern()

    def _maybe_scan_selector(self):
        """From the current position, consume a directly-adjacent '@name'/'@name()'
        selector suffix if present (no whitespace allowed before '@')."""
        if self._peek_char() == '@':
            selector, self.i = _scan_selector_ref(self.text, self.i)
            return selector
        return None

    def parse_set_literal(self):
        self._expect('{')
        names = []
        self._skip_ws()
        if self._peek_char() != '}':
            names.append(self._read_name())
            self._skip_ws()
            while self._peek_char() == ',':
                self.i += 1
                self._skip_ws()
                names.append(self._read_name())
                self._skip_ws()
        self._expect('}')
        return SetLiteral(names)

    def _read_name(self):
        m = re.match(r'[^\s,}]+', self.text[self.i:])
        if not m:
            raise ValueError(f"Expected a name in {self.text!r} at position {self.i}")
        self.i += m.end()
        return m.group(0)

    def parse_pattern(self):
        start = self.i
        paren_depth = 0
        brace_depth = 0
        while self.i < self.n:
            c = self.text[self.i]
            if paren_depth == 0 and brace_depth == 0 and (c.isspace() or c == '@'):
                break
            if c == '(':
                paren_depth += 1
            elif c == ')':
                if paren_depth == 0:
                    break
                paren_depth -= 1
            elif c == '{':
                brace_depth += 1
            elif c == '}':
                if brace_depth == 0:
                    break
                brace_depth -= 1
            self.i += 1
        if self.i == start and self._peek_char() != '@':
            # Zero pattern chars is only valid when going straight into a
            # selector (e.g. '@ohe_drop_first()' == "all columns, then select").
            raise ValueError(f"Unexpected character {self.text[start]!r} in {self.text!r} at position {start}")
        regex = self.text[start:self.i]
        return Pattern(regex, self._maybe_scan_selector())


def parse(dsl_string):
    """Parse a DSL string into an AST node."""
    parser = _Parser(dsl_string)
    node = parser.parse_expr()
    parser._skip_ws()
    if parser.i != parser.n:
        raise ValueError(f"Unexpected trailing input in {dsl_string!r} at position {parser.i}")
    return node


def _apply_selector(selector, matched, data, processor):
    if selector is None:
        return matched
    return resolve_selector(selector, data.select_columns(matched), processor)


def eval_expr(node, data, processor=None):
    """Evaluate an AST node against ``data`` (a ``DataWrapper`` exposing
    ``get_columns()``, and — for dtype-based selectors like ``@numeric`` —
    ``select_by_dtype()``), returning the selected column names (order
    preserved). ``Namespace`` is only valid as a top-level segment, not
    inside an already-namespaced expression."""
    columns = data.get_columns()
    if isinstance(node, Star):
        return _apply_selector(node.selector, list(columns), data, processor)
    if isinstance(node, slice):
        return list(columns[node])
    if isinstance(node, SetLiteral):
        missing = [n for n in node.names if n not in columns]
        if missing:
            raise ValueError(f"Unknown column(s): {missing}")
        return _apply_selector(node.selector, list(node.names), data, processor)
    if isinstance(node, Pattern):
        matched = [c for c in columns if re.match(node.regex, str(c))]
        return _apply_selector(node.selector, matched, data, processor)
    if isinstance(node, BinOp):
        left = eval_expr(node.left, data, processor)
        right = eval_expr(node.right, data, processor)
        if node.op == '+':
            seen = set(left)
            extra = [c for c in right if c not in seen and not seen.add(c)]
            return left + extra
        if node.op == '-':
            right_set = set(right)
            return [c for c in left if c not in right_set]
        if node.op == '&':
            right_set = set(right)
            return [c for c in left if c in right_set]
    if isinstance(node, Namespace):
        raise ValueError(f"Namespace '{node.name}:' is not valid inside another namespace")
    raise TypeError(f"Cannot evaluate {type(node).__name__}")


def flatten_plus(node):
    """Split the top-level '+'-chain into ordered segments (only '+' splits;
    any other node — including '-'/'&' BinOps — is one opaque segment)."""
    if isinstance(node, BinOp) and node.op == '+':
        return flatten_plus(node.left) + flatten_plus(node.right)
    return [node]


def iter_segments(dsl_string):
    """Parse and split into ``(node_name, expr)`` pairs — ``node_name`` is
    ``None`` for a bare/DataSource segment, or the namespace name."""
    for seg in flatten_plus(parse(dsl_string)):
        if isinstance(seg, Namespace):
            yield seg.name, seg.expr
        else:
            yield None, seg


def referenced_nodes(dsl_string):
    """Return the set of node names referenced as top-level segments —
    ``None`` for the DataSource, or a stage node name per ``name:(...)`` block."""
    return {name for name, _ in iter_segments(dsl_string)}


def unparse(node):
    """Render an AST node back to DSL text (used for display/diffing)."""
    if isinstance(node, Star):
        return '*' + (node.selector or '')
    if isinstance(node, slice):
        start = '' if node.start is None else str(node.start)
        stop = '' if node.stop is None else str(node.stop)
        return f'{start}:{stop}'
    if isinstance(node, SetLiteral):
        return '{' + ', '.join(node.names) + '}' + (node.selector or '')
    if isinstance(node, Pattern):
        return node.regex + (node.selector or '')
    if isinstance(node, Namespace):
        return f'{node.name}:({unparse(node.expr)})'
    if isinstance(node, BinOp):
        return f'{unparse(node.left)} {node.op} {unparse(node.right)}'
    raise TypeError(f"Cannot unparse {type(node).__name__}")


def validate_edges(dsl_string, pipeline):
    """Structural-only validation: syntax, namespace references (must exist
    as stage nodes), and the "only '+' at the top level" rule.

    Deliberately does **not** resolve columns/variables — a node's edges are
    only ever expanded into an actual column list lazily, when the Processor
    processes real data (see ``_flow.py.get_data`` / :func:`eval_expr`), not
    at ``set_grp``/``set_node`` time.
    """
    for seg in flatten_plus(parse(dsl_string)):
        if isinstance(seg, Namespace):
            name = seg.name
            if name not in pipeline.nodes:
                raise ValueError(f"Edge namespace '{name}:' does not reference an existing node")
            grp = pipeline.grps[pipeline.nodes[name].grp]
            if grp.role != 'stage':
                raise ValueError(f"Edge namespace '{name}:' must be a stage node, got '{grp.role}'")
        else:
            _check_no_namespace(seg)


def _check_no_namespace(node):
    if isinstance(node, Namespace):
        raise ValueError(
            "Cross-namespace '-'/'&' is not supported at the top level; "
            "wrap the namespace block so only '+' joins top-level segments"
        )
    if isinstance(node, BinOp):
        _check_no_namespace(node.left)
        _check_no_namespace(node.right)
