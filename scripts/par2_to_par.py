#!/usr/bin/env python3
"""Expand an Abacus-style .par2 file into a plain .par file that ParseHeader can read.

Fallback for Abacus's own converter, Abacus/GenParam.py (abacus-public). run.sh uses
GenParam when $ABACUS points to an Abacus checkout, and this script otherwise. It follows
GenParam's rules so both give the same parameters:

  - comments start at any '#' that is not '#include' (even inside quotes, as in GenParam)
  - #include "file"     path relative to the including file; $VAR$ expands environment variables
  - @Name@              value of another parameter, resolved after the whole file is read;
                        @_Name@ is an optional reference to Name and becomes "" if undefined
  - values              tried in GenParam's order: the space-separated items as a vector (one
                        item gives a scalar), then an expression (e.g. 256**3, @L@*2), then a
                        string; a value that is none of these is an error. Quote strings:
                        like GenParam, 'Key = word' gives " word" (with the leading space)
  - overrides (-D/-s)   win over anything in the file, like GenParam.makeInput(**keywords)
  - output              sorted 'Key = value' lines; strings double-quoted

Unlike GenParam, expressions are evaluated with a restricted evaluator (arithmetic, numbers,
strings, int/float/min/max/abs/round), not Python's eval.

Usage:
  par2_to_par.py IN.par2 OUT.par [-D Key=expr ...] [-s Key=string ...]
"""

import argparse
import ast
import operator
import os
import re
import shlex
import sys

COMMENT_RE = re.compile(r"#(?!include)")
VAR_RE = re.compile(r"@(\w+)@")

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"int": int, "float": float, "min": min, "max": max, "abs": abs, "round": round}


class Par2Error(Exception):
    pass


def _eval_node(node, names):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, names)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left, names), _eval_node(node.right, names))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand, names))
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_eval_node(e, names) for e in node.elts)
    if isinstance(node, ast.Name) and node.id in names:
        return names[node.id]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
            and not node.keywords):
        return _FUNCS[node.func.id](*(_eval_node(a, names) for a in node.args))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def safe_eval(text, names):
    return _eval_node(ast.parse(text.lstrip(" \t"), mode="eval"), names)  # eval() also skips these


def shellreplace(line, where):
    """Replace each $VAR$ with the environment variable VAR (GenParam.shellreplace)."""
    while "$" in line:
        start = line.index("$")
        end = line.find("$", start + 1)
        if end < 0:
            raise Par2Error(f"{where}: unmatched '$' in: {line}")
        var = line[start + 1:end]
        if var not in os.environ:
            raise Par2Error(f"{where}: environment variable {var} is not defined")
        line = line[:start] + os.environ[var] + line[end + 1:]
    return line


_EVAL_ERRORS = (ValueError, SyntaxError, TypeError, ArithmeticError)


def parse_value(text, names, where="value"):
    """GenParam._parse_assignment, with safe_eval in place of eval: the shlex-split items as
    a tuple "(a,b,...)" (one item gives a scalar, none gives ()), then the whole value as an
    expression, then the value as a string."""
    try:
        return safe_eval("(" + ",".join(shlex.split(text)) + ")", names)
    except _EVAL_ERRORS:
        pass
    try:
        return safe_eval(text, names)
    except _EVAL_ERRORS:
        pass
    try:
        return safe_eval('"' + text + '"', names)
    except _EVAL_ERRORS:
        raise Par2Error(f"{where}: could not parse value: {text}")


def resolve_refs(text, values, where):
    """Turn @Name@ into an evaluator variable; returns (text, names) or None if not ready.
    As in GenParam.varreplace, @_Name@ is an optional reference to Name and becomes "" if
    Name is undefined."""
    names = {}

    def sub(m):
        key = m.group(1)
        optional = key.startswith("_")
        if optional:
            key = key[1:]
        if key in values:
            names[f"__v_{key}"] = values[key]
            return f"__v_{key}"
        if optional:
            return '""'
        raise KeyError(key)

    try:
        return VAR_RE.sub(sub, text), names
    except KeyError:
        return None


def read_par2(path, values, fixed, deferred, stack=()):
    path = os.path.abspath(path)
    if path in stack:
        raise Par2Error(f"#include cycle: {' -> '.join(stack + (path,))}")
    with open(path) as f:
        lines = f.readlines()
    for lineno, raw in enumerate(lines, 1):
        where = f"{path}:{lineno}"
        line = raw.rstrip("\n")
        if "\x02" in line:  # ParseHeader end-of-header token
            break
        m = COMMENT_RE.search(line)
        if m is not None:
            line = line[:m.start()]
        line = shellreplace(line, where).strip()
        if not line:
            continue
        if line.startswith("#include"):
            inc = line[len("#include"):].strip().strip("\"'")
            if not os.path.isabs(inc):
                inc = os.path.join(os.path.dirname(path), inc)
            read_par2(inc, values, fixed, deferred, stack + (path,))
            continue
        if "=" not in line:
            raise Par2Error(f"{where}: expected 'Key = value': {line}")
        key, text = line.split("=", 1)
        key = key.strip()  # text keeps its leading space, as in GenParam (see module doc)
        if not re.fullmatch(r"\w+", key):
            raise Par2Error(f"{where}: bad parameter name '{key}'")
        if key in fixed:
            continue
        if "@" in text:
            deferred[key] = (text, where)
            values.pop(key, None)
        else:
            values[key] = parse_value(text, {}, where)
            deferred.pop(key, None)


def resolve_deferred(values, deferred):
    while deferred:
        progress = False
        for key, (text, where) in list(deferred.items()):
            r = resolve_refs(text, values, where)
            if r is None:
                continue
            expr, names = r
            values[key] = parse_value(expr, names, where)
            del deferred[key]
            progress = True
        if not progress:
            missing = sorted({m for t, _ in deferred.values() for m in VAR_RE.findall(t)
                              if m not in values and not m.startswith("_")})
            where = "; ".join(w for _, w in deferred.values())
            raise Par2Error(f"undefined @references@ {missing} at {where}")


def tostr(v):
    if isinstance(v, str):
        return v if '"' in v else f'"{v}"'
    if isinstance(v, (tuple, list)):
        return " ".join(tostr(i) for i in v)
    return repr(v)


def convert(in_path, out_path, overrides):
    values = dict(overrides)
    deferred = {}
    read_par2(in_path, values, set(overrides), deferred)
    resolve_deferred(values, deferred)
    with open(out_path, "w") as f:
        for key in sorted(values):
            if values[key] is None:
                continue
            f.write(f"{key} = {tostr(values[key])}\n")
        f.write(f"#created by par2_to_par.py from {os.path.abspath(in_path)}\n")
        f.write("\x02\n")  # ParseHeader end-of-header token, as GenParam writes


def _override(s, as_string):
    if "=" not in s:
        raise Par2Error(f"override must be Key=Value, got {s!r}")
    key, val = (t.strip() for t in s.split("=", 1))
    return key, (val if as_string else parse_value(val, {}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help=".par2 file")
    ap.add_argument("output", help=".par file to write")
    ap.add_argument("-D", dest="exprs", action="append", default=[], metavar="Key=expr",
                    help="override, parsed like a .par2 value")
    ap.add_argument("-s", dest="strings", action="append", default=[], metavar="Key=string",
                    help="override with a literal string (use for paths)")
    args = ap.parse_args()
    try:
        overrides = dict(_override(s, False) for s in args.exprs)
        overrides.update(_override(s, True) for s in args.strings)
        convert(args.input, args.output, overrides)
    except (Par2Error, OSError) as e:
        sys.exit(f"par2_to_par: {e}")


if __name__ == "__main__":
    main()
