"""Composite analysis tools that aggregate multiple data sources."""

from __future__ import annotations

from collections import defaultdict
from typing import Annotated, Any, TypedDict

from .rpc import tool, unsafe
from .sync import idasync, tool_timeout, IDAError
from .utils import (
    parse_address,
    resolve_addr,
    get_function,
    get_prototype,
    get_callees,
    get_callers,
    get_all_xrefs,
    get_all_comments,
    extract_function_strings,
    extract_function_constants,
    get_stack_frame_variables_internal,
    decompile_function_safe,
    get_assembly_lines,
    normalize_list_input,
)

# Max decompile lines before truncation.
_DECOMPILE_LINE_CAP = 100
# Max strings/constants returned in compact mode.
_TOP_STRINGS = 10
_TOP_CONSTANTS = 10
# Constants filtered out of extract_function_constants results.
_BORING_CONSTANTS = frozenset({0, 1, -1, 0xFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF})


class BasicBlockSummary(TypedDict):
    count: int
    cyclomatic_complexity: int


class AnalyzeFunctionResult(TypedDict, total=False):
    addr: str
    name: str
    prototype: str | None
    size: int
    decompiled: str | None
    decompile_truncated: int
    assembly: str | None
    strings: list[str]
    constants: list[dict[str, Any]]
    callees: list[str]
    callers: list[str]
    xrefs: dict[str, Any]
    comments: dict[str, Any]
    basic_blocks: BasicBlockSummary
    error: str | None


class ComponentFunctionSummary(TypedDict, total=False):
    addr: str
    name: str
    prototype: str | None
    size: int
    callees: list[str]
    strings: list[str]
    basic_blocks: int
    complexity: int
    error: str


ComponentGraphEdge = TypedDict(
    "ComponentGraphEdge",
    {"from": str, "to": str, "name": str},
)


class InternalCallGraph(TypedDict):
    nodes: list[str]
    edges: list[ComponentGraphEdge]


class SharedGlobalInfo(TypedDict):
    addr: str
    name: str
    accessed_by: list[str]


class AnalyzeComponentResult(TypedDict, total=False):
    functions: list[ComponentFunctionSummary]
    internal_call_graph: InternalCallGraph
    shared_globals: list[SharedGlobalInfo]
    interface_functions: list[str]
    internal_only: list[str]
    string_usage: dict[str, list[str]]
    error: str


class DiffBeforeAfterResult(TypedDict, total=False):
    before: str | None
    after: str | None
    action_applied: str
    changes_detected: bool
    error: str


class TraceDataFlowNode(TypedDict):
    addr: str
    func: str | None
    instruction: str | None
    type: str
    name: str | None
    depth: int


TraceDataFlowEdge = TypedDict(
    "TraceDataFlowEdge",
    {"from": str, "to": str, "type": str},
)


class TraceDataFlowResult(TypedDict, total=False):
    start: str
    direction: str
    depth_reached: int
    nodes: list[TraceDataFlowNode]
    edges: list[TraceDataFlowEdge]
    error: str


class PortingDependency(TypedDict, total=False):
    addr: str
    name: str
    kind: str


class PortingBundleResult(TypedDict, total=False):
    addr: str
    name: str
    prototype: str | None
    size: int
    decompiled: str | None
    decompile_truncated: int
    assembly: str | None
    stack_frame: list[dict[str, Any]]
    strings: list[str]
    constants: list[dict[str, Any]]
    callees: list[PortingDependency]
    callers: list[PortingDependency]
    xref_summary: dict[str, int]
    comments: dict[str, Any]
    sdk_stub: str | None
    notes: list[str]
    error: str | None


# ---------------------------------------------------------------------------
# Internal helpers (no @tool — called from within @idasync context)
# ---------------------------------------------------------------------------

# Local alias kept for backwards compatibility within this module.
_resolve_addr = resolve_addr


def _basic_block_info(ea: int) -> BasicBlockSummary:
    """Return block count and cyclomatic complexity for the function at *ea*."""
    import idaapi

    func = idaapi.get_func(ea)
    if func is None:
        return {"count": 0, "cyclomatic_complexity": 0}

    fc = idaapi.FlowChart(func)
    nodes = 0
    edges = 0
    for block in fc:
        nodes += 1
        for _ in block.succs():
            edges += 1

    return {"count": nodes, "cyclomatic_complexity": edges - nodes + 2}


def _filter_constants(raw: list[dict], limit: int = _TOP_CONSTANTS) -> list[dict]:
    """Drop boring constants, return top N by absolute value."""
    out = []
    for c in raw:
        val = c.get("value", 0)
        if not isinstance(val, int):
            continue
        if abs(val) < 0x100 or val in _BORING_CONSTANTS:
            continue
        out.append(c)
    out.sort(key=lambda c: abs(c.get("value", 0)) if isinstance(c.get("value"), int) else 0, reverse=True)
    return out[:limit]


def _cap_decompile(code: str | None) -> tuple[str | None, int | None]:
    """Cap decompiled output at _DECOMPILE_LINE_CAP lines.
    Returns (possibly_truncated_code, total_lines_or_None)."""
    if code is None:
        return None, None
    lines = code.split("\n")
    total = len(lines)
    if total <= _DECOMPILE_LINE_CAP:
        return code, None  # not truncated
    truncated = "\n".join(lines[:_DECOMPILE_LINE_CAP])
    return truncated, total


def _compact_strings(raw: list[dict], limit: int = _TOP_STRINGS) -> list[str]:
    """Return just the string values, deduplicated, capped at limit."""
    seen: set[str] = set()
    out: list[str] = []
    for s in raw:
        val = s.get("value") or s.get("string", "")
        if val and val not in seen:
            seen.add(val)
            out.append(val)
            if len(out) >= limit:
                break
    return out


def _compact_callees(raw: list[dict]) -> list[str]:
    """Return just callee names/addresses as strings."""
    return [c.get("name") or c.get("addr", "?") for c in raw]


def _limit_text_lines(text: str | None, max_lines: int) -> tuple[str | None, int | None]:
    if text is None:
        return None, None
    lines = text.split("\n")
    total = len(lines)
    if total <= max_lines:
        return text, None
    return "\n".join(lines[:max_lines]), total


def _dependency_rows(raw: list[dict], limit: int) -> list[PortingDependency]:
    rows: list[PortingDependency] = []
    for item in raw[:limit]:
        rows.append(
            {
                "addr": str(item.get("addr", "")),
                "name": str(item.get("name") or item.get("addr", "")),
                "kind": str(item.get("type", "function")),
            }
        )
    return rows


def _sanitize_identifier(name: str) -> str:
    out = []
    for char in name:
        if char.isalnum() or char == "_":
            out.append(char)
        else:
            out.append("_")
    sanitized = "".join(out).strip("_")
    if not sanitized:
        sanitized = "reconstructed_function"
    if sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


def _make_sdk_stub(name: str, prototype: str | None, pseudocode: str | None) -> str:
    fn_name = _sanitize_identifier(name or "reconstructed_function")
    signature = prototype.strip() if prototype else f"void {fn_name}(void)"
    if prototype and "(" in signature:
        head, _, tail = signature.partition("(")
        signature = f"{head.strip()}({tail}"
    elif not prototype:
        signature = f"void {fn_name}(void)"

    return_head = signature.split("(", 1)[0].strip()
    return_type = return_head[: -len(fn_name)].strip() if return_head.endswith(fn_name) else ""

    body_lines = [
        signature,
        "{",
        "    // Reconstructed from IDA MCP output for Source SDK style reimplementation.",
        "    // Port the control flow below, replace engine-only helpers with SDK equivalents,",
        "    // and validate behavior against the original binary before shipping.",
    ]
    if pseudocode:
        body_lines.append("    /*")
        for line in pseudocode.split("\n")[:20]:
            body_lines.append(f"    {line}")
        body_lines.append("    */")
    body_lines.append('    AssertMsg(false, "Reconstructed stub still needs logic ported from the original binary");')
    if return_type and return_type != "void":
        body_lines.append("    return {};")
    body_lines.append("}")
    return "\n".join(body_lines)


def _analyze_function_internal(
    ea: int, *, include_asm: bool = False
) -> AnalyzeFunctionResult:
    """Core analysis logic — must be called from an @idasync context.

    Returns a compact response by default: decompilation capped at 100 lines,
    top 10 strings as values only, top 10 non-trivial constants, no disassembly.
    Pass include_asm=True to include full disassembly."""
    import idaapi

    result: dict = {"addr": hex(ea), "error": None}

    try:
        func = idaapi.get_func(ea)
        if func is None:
            result["error"] = f"No function at {hex(ea)}"
            return result

        result["name"] = idaapi.get_func_name(ea) or ""
        result["prototype"] = get_prototype(func)
        result["size"] = func.end_ea - func.start_ea

        # Decompilation — capped at _DECOMPILE_LINE_CAP lines.
        try:
            raw_code = decompile_function_safe(ea)
            code, total_lines = _cap_decompile(raw_code)
            result["decompiled"] = code
            if total_lines is not None:
                result["decompile_truncated"] = total_lines
        except Exception:
            result["decompiled"] = None

        # Assembly — opt-in only.
        if include_asm:
            try:
                result["assembly"] = get_assembly_lines(ea)
            except Exception:
                result["assembly"] = None

        # Strings — top 10 values only.
        result["strings"] = _compact_strings(extract_function_strings(ea))
        # Constants — top 10 non-trivial.
        result["constants"] = _filter_constants(extract_function_constants(ea))
        # Callees/callers — names only.
        result["callees"] = _compact_callees(get_callees(hex(ea)))
        result["callers"] = _compact_callees(get_callers(hex(ea)))
        result["xrefs"] = get_all_xrefs(ea)
        result["comments"] = get_all_comments(ea)
        result["basic_blocks"] = _basic_block_info(ea)

    except Exception as exc:
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Tool 1 — analyze_function
# ---------------------------------------------------------------------------


@tool
@idasync
@tool_timeout(120.0)
def analyze_function(
    addr: Annotated[str, "Function address or name"],
    include_asm: Annotated[bool, "Include full disassembly (default: false, saves tokens)"] = False,
) -> AnalyzeFunctionResult:
    """Compact single-function analysis: pseudocode, strings, constants, callers, callees, xrefs, blocks."""

    try:
        ea = _resolve_addr(addr)
    except IDAError as exc:
        return {"addr": addr, "error": str(exc)}

    return _analyze_function_internal(ea, include_asm=include_asm)


# ---------------------------------------------------------------------------
# Tool 2 — analyze_component
# ---------------------------------------------------------------------------


@tool
@idasync
@tool_timeout(180.0)
def analyze_component(
    addrs: Annotated[list[str] | str, "Function addresses (comma-separated or list)"],
) -> AnalyzeComponentResult:
    """Analyze related functions as a group: per-function summaries, internal call graph, shared data."""

    import idaapi
    import idautils

    raw = normalize_list_input(addrs)
    if not raw:
        return {"error": "Empty address list"}

    ea_map: dict[int, str] = {}
    for a in raw:
        try:
            ea_map[_resolve_addr(a)] = a
        except IDAError:
            return {"error": f"Cannot resolve address: {a!r}"}

    ea_set = set(ea_map.keys())

    # --- Per-function COMPACT summary (no decompile, no disasm) ---
    functions: list[dict] = []
    for ea in ea_set:
        func = idaapi.get_func(ea)
        if func is None:
            functions.append({"addr": hex(ea), "error": "No function"})
            continue
        name = idaapi.get_func_name(ea) or ""
        strings_raw = extract_function_strings(ea)
        top_strings = _compact_strings(strings_raw, limit=5)
        callee_list = _compact_callees(get_callees(hex(ea)))
        bb = _basic_block_info(ea)
        functions.append({
            "addr": hex(ea),
            "name": name,
            "prototype": get_prototype(func),
            "size": func.end_ea - func.start_ea,
            "callees": callee_list,
            "strings": top_strings,
            "basic_blocks": bb["count"],
            "complexity": bb["cyclomatic_complexity"],
        })

    # --- Internal call graph ---
    nodes = [hex(ea) for ea in ea_set]
    edges: list[dict] = []
    for ea in ea_set:
        for callee in (get_callees(hex(ea)) or []):
            callee_ea = callee.get("addr")
            if isinstance(callee_ea, str):
                try:
                    callee_ea = int(callee_ea, 16)
                except (ValueError, TypeError):
                    continue
            if callee_ea in ea_set:
                edges.append({
                    "from": hex(ea),
                    "to": hex(callee_ea),
                    "name": callee.get("name", ""),
                })

    # --- Shared globals ---
    func_globals: dict[int, set[int]] = {}
    for ea in ea_set:
        globals_accessed: set[int] = set()
        func = idaapi.get_func(ea)
        if func is None:
            func_globals[ea] = globals_accessed
            continue
        for head in idautils.Heads(func.start_ea, func.end_ea):
            for xref in idautils.XrefsFrom(head, 0):
                if xref.iscode:
                    continue
                ref_func = idaapi.get_func(xref.to)
                if ref_func is None and idaapi.is_loaded(xref.to):
                    globals_accessed.add(xref.to)
        func_globals[ea] = globals_accessed

    global_refcount: dict[int, list[str]] = defaultdict(list)
    for ea, gset in func_globals.items():
        fname = idaapi.get_func_name(ea) or hex(ea)
        for g in gset:
            global_refcount[g].append(fname)

    shared_globals = []
    for g_ea, accessors in sorted(global_refcount.items()):
        if len(accessors) >= 2:
            shared_globals.append({
                "addr": hex(g_ea),
                "name": idaapi.get_name(g_ea) or hex(g_ea),
                "accessed_by": sorted(accessors),
            })

    # --- Interface vs internal ---
    interface_functions: list[str] = []
    internal_only: list[str] = []
    for ea in ea_set:
        callers = get_callers(hex(ea))
        has_external = False
        for c in (callers or []):
            caller_addr = c.get("addr") or c.get("start_ea")
            if isinstance(caller_addr, str):
                try:
                    caller_addr = int(caller_addr, 16)
                except (ValueError, TypeError):
                    has_external = True
                    break
            if caller_addr not in ea_set:
                has_external = True
                break
        if has_external:
            interface_functions.append(hex(ea))
        else:
            internal_only.append(hex(ea))

    # --- String usage across functions ---
    string_funcs: dict[str, set[str]] = defaultdict(set)
    for ea in ea_set:
        fname = idaapi.get_func_name(ea) or hex(ea)
        for s in (extract_function_strings(ea) or []):
            sval = s.get("value") or s.get("string", "")
            if sval:
                string_funcs[sval].add(fname)

    string_usage = {
        s: sorted(fnames)
        for s, fnames in sorted(string_funcs.items())
        if len(fnames) >= 2
    }

    return {
        "functions": functions,
        "internal_call_graph": {"nodes": nodes, "edges": edges},
        "shared_globals": shared_globals,
        "interface_functions": interface_functions,
        "internal_only": internal_only,
        "string_usage": string_usage,
    }


# ---------------------------------------------------------------------------
# Tool 3 — diff_before_after
# ---------------------------------------------------------------------------

_VALID_ACTIONS = frozenset({"rename_func", "set_type", "set_comment"})



@tool
@unsafe
@idasync
@tool_timeout(120.0)
def diff_before_after(
    addr: Annotated[str, "Function address"],
    action: Annotated[str, "Action: 'rename_func', 'set_type', 'set_comment'"],
    action_args: Annotated[dict, "Arguments for the action"],
) -> DiffBeforeAfterResult:
    """Rename a function, set its type, or add a comment, and immediately see the
    before/after decompilation side by side. Use this instead of calling rename
    then decompile separately when you want to verify that a rename or type change
    actually improved readability. Actions: 'rename_func' (action_args: {name: str}),
    'set_type' (action_args: {type: str}), 'set_comment' (action_args: {comment: str}).
    Returns {before, after, action_applied, changes_detected}. Especially useful
    during batch renaming to confirm each change had the intended effect."""

    import idaapi
    import ida_hexrays
    import ida_typeinf

    if action not in _VALID_ACTIONS:
        return {"error": f"Invalid action {action!r}. Must be one of: {', '.join(sorted(_VALID_ACTIONS))}"}

    try:
        ea = _resolve_addr(addr)
    except IDAError as exc:
        return {"error": str(exc)}

    func = idaapi.get_func(ea)
    if func is None:
        return {"error": f"No function at {hex(ea)}"}

    # --- Before ---
    before = decompile_function_safe(ea)

    # --- Apply action ---
    applied: str
    try:
        if action == "rename_func":
            name = action_args.get("name")
            if not name:
                return {"error": "action_args must contain 'name'"}
            ok = idaapi.set_name(ea, name, idaapi.SN_CHECK)
            if not ok:
                return {"error": f"set_name failed for {name!r}"}
            applied = f"Renamed to {name!r}"

        elif action == "set_type":
            type_str = action_args.get("type")
            if not type_str:
                return {"error": "action_args must contain 'type'"}
            from .api_types import _parse_function_tinfo
            try:
                tif = _parse_function_tinfo(type_str)
            except ValueError:
                return {"error": f"Failed to parse type: {type_str!r}"}
            ok = ida_typeinf.apply_tinfo(ea, tif, ida_typeinf.TINFO_DEFINITE)
            if not ok:
                return {"error": f"apply_tinfo failed for {type_str!r}"}
            applied = f"Set type to {type_str!r}"

        elif action == "set_comment":
            comment = action_args.get("comment")
            if comment is None:
                return {"error": "action_args must contain 'comment'"}
            idaapi.set_cmt(ea, comment, False)
            applied = f"Set comment: {comment!r}"

        else:
            return {"error": f"Unhandled action {action!r}"}
    except Exception as exc:
        return {"error": f"Action {action!r} failed: {exc}"}

    # --- After (invalidate Hex-Rays cache so we see the change) ---
    ida_hexrays.mark_cfunc_dirty(ea)
    after = decompile_function_safe(ea)

    return {
        "before": before,
        "after": after,
        "action_applied": applied,
        "changes_detected": before != after,
    }


# ---------------------------------------------------------------------------
# Tool 4 — trace_data_flow
# ---------------------------------------------------------------------------

_MAX_TRACE_NODES = 200
_MAX_TRACE_EDGES = 500



@tool
@idasync
@tool_timeout(120.0)
def trace_data_flow(
    addr: Annotated[str, "Starting address"],
    direction: Annotated[str, "'forward' (xrefs from) or 'backward' (xrefs to)"] = "forward",
    max_depth: Annotated[int, "Maximum traversal depth"] = 5,
) -> TraceDataFlowResult:
    """Follow cross-references from or to an address, automatically traversing
    multiple hops. Use 'forward' to see where data flows TO (xrefs-from), or
    'backward' to see where data flows FROM (xrefs-to). At each node in the
    traversal, returns the function name, instruction, and whether it's code or
    data. Use this when you find an interesting string, constant, or global and
    want to understand every code path that touches it without manually chaining
    xrefs_to calls. Do not use for call graph traversal — use callgraph for that.
    max_depth controls how many hops to follow (default 5, max 20)."""

    import idaapi
    import idautils
    import idc
    from collections import deque

    if direction not in ("forward", "backward"):
        return {"error": f"direction must be 'forward' or 'backward', got {direction!r}"}

    try:
        start_ea = _resolve_addr(addr)
    except IDAError as exc:
        return {"error": str(exc)}

    if max_depth < 1:
        max_depth = 1
    if max_depth > 20:
        max_depth = 20

    visited: set[int] = set()
    nodes: list[dict] = []
    edges: list[dict] = []
    depth_reached = 0

    # BFS queue: (ea, depth)
    queue: deque[tuple[int, int]] = deque()
    queue.append((start_ea, 0))
    visited.add(start_ea)

    while queue and len(nodes) < _MAX_TRACE_NODES:
        ea, depth = queue.popleft()
        if depth > max_depth:
            continue
        if depth > depth_reached:
            depth_reached = depth

        # Build node info.
        func = idaapi.get_func(ea)
        func_name = idaapi.get_func_name(ea) if func else None
        insn_text = idc.GetDisasm(ea) if idaapi.is_loaded(ea) else None

        # Determine if this address references a global/string.
        name_at = idaapi.get_name(ea)
        node_type = "code"
        if func is None and idaapi.is_loaded(ea):
            node_type = "data"

        nodes.append({
            "addr": hex(ea),
            "func": func_name,
            "instruction": insn_text,
            "type": node_type,
            "name": name_at if name_at else None,
            "depth": depth,
        })

        if depth >= max_depth:
            continue

        # Follow xrefs in the requested direction.
        if direction == "forward":
            xrefs = list(idautils.XrefsFrom(ea, 0))
        else:
            xrefs = list(idautils.XrefsTo(ea, 0))

        for xref in xrefs:
            if len(edges) >= _MAX_TRACE_EDGES:
                break
            target = xref.to if direction == "forward" else xref.frm
            # Classify xref type.
            xtype = "code" if xref.iscode else "data"

            edges.append({
                "from": hex(ea) if direction == "forward" else hex(target),
                "to": hex(target) if direction == "forward" else hex(ea),
                "type": xtype,
            })

            if target not in visited and len(nodes) + len(queue) < _MAX_TRACE_NODES:
                visited.add(target)
                queue.append((target, depth + 1))

    return {
        "start": hex(start_ea),
        "direction": direction,
        "depth_reached": depth_reached,
        "nodes": nodes,
        "edges": edges,
    }


# ---------------------------------------------------------------------------
# Tool 5 — build_porting_bundle
# ---------------------------------------------------------------------------


@tool
@idasync
@tool_timeout(180.0)
def build_porting_bundle(
    addr: Annotated[str, "Function address or name to package for source-porting work"],
    include_asm: Annotated[bool, "Include assembly text alongside decompiled output for validation"] = True,
    max_decompile_lines: Annotated[int, "Maximum decompiled lines to include in the bundle"] = 220,
    max_asm_lines: Annotated[int, "Maximum assembly lines to include in the bundle"] = 160,
    max_stack_vars: Annotated[int, "Maximum stack variables to include in the porting summary"] = 48,
    max_dependencies: Annotated[int, "Maximum callees and callers to include in the dependency summary"] = 24,
) -> PortingBundleResult:
    """Build a reconstruction bundle for porting a function into a codebase such as Source SDK 2013. The result combines prototype, capped pseudocode, optional assembly, stack variables, strings, constants, dependency summaries, and a generated C++ stub skeleton so an agent can start replacing an engine routine instead of re-querying five separate tools. Use this when you want to overwrite or reimplement engine/game functions in your own project while keeping the original behavior traceable back to IDA."""
    import idaapi

    max_decompile_lines = max(20, min(int(max_decompile_lines), 500))
    max_asm_lines = max(20, min(int(max_asm_lines), 400))
    max_stack_vars = max(0, min(int(max_stack_vars), 128))
    max_dependencies = max(1, min(int(max_dependencies), 128))

    try:
        ea = _resolve_addr(addr)
    except IDAError as exc:
        return {"addr": addr, "error": str(exc)}

    func = idaapi.get_func(ea)
    if func is None:
        return {"addr": hex(ea), "error": f"No function at {hex(ea)}"}

    start_ea = func.start_ea
    result: PortingBundleResult = {
        "addr": hex(start_ea),
        "name": idaapi.get_func_name(start_ea) or "",
        "prototype": get_prototype(func),
        "size": func.end_ea - func.start_ea,
        "strings": [],
        "constants": [],
        "callees": [],
        "callers": [],
        "stack_frame": [],
        "xref_summary": {"to": 0, "from": 0},
        "comments": {},
        "notes": [],
        "error": None,
    }

    try:
        raw_code = decompile_function_safe(start_ea)
        code, total_lines = _limit_text_lines(raw_code, max_decompile_lines)
        result["decompiled"] = code
        if total_lines is not None:
            result["decompile_truncated"] = total_lines
            result["notes"].append(
                f"Decompiled output truncated to {max_decompile_lines} lines from {total_lines} total lines."
            )
    except Exception as exc:
        result["decompiled"] = None
        result["notes"].append(f"Decompiler failed: {exc}")

    if include_asm:
        try:
            asm_lines = get_assembly_lines(start_ea)
            asm_text, total_asm_lines = _limit_text_lines(asm_lines, max_asm_lines)
            result["assembly"] = asm_text
            if total_asm_lines is not None:
                result["notes"].append(
                    f"Assembly output truncated to {max_asm_lines} lines from {total_asm_lines} total lines."
                )
        except Exception as exc:
            result["assembly"] = None
            result["notes"].append(f"Assembly export failed: {exc}")
    else:
        result["assembly"] = None

    try:
        result["stack_frame"] = get_stack_frame_variables_internal(start_ea, False)[:max_stack_vars]
    except Exception as exc:
        result["notes"].append(f"Stack frame extraction failed: {exc}")

    try:
        result["strings"] = _compact_strings(extract_function_strings(start_ea), limit=24)
    except Exception as exc:
        result["notes"].append(f"String extraction failed: {exc}")

    try:
        result["constants"] = _filter_constants(extract_function_constants(start_ea), limit=24)
    except Exception as exc:
        result["notes"].append(f"Constant extraction failed: {exc}")

    try:
        callees = get_callees(hex(start_ea)) or []
        callers = get_callers(hex(start_ea)) or []
        result["callees"] = _dependency_rows(callees, max_dependencies)
        result["callers"] = _dependency_rows(callers, max_dependencies)
    except Exception as exc:
        result["notes"].append(f"Dependency extraction failed: {exc}")

    try:
        xrefs = get_all_xrefs(start_ea)
        result["comments"] = get_all_comments(start_ea)
        result["xref_summary"] = {
            "to": len(list(xrefs.get("to", []))),
            "from": len(list(xrefs.get("from", []))),
        }
    except Exception as exc:
        result["notes"].append(f"Xref/comment extraction failed: {exc}")

    result["sdk_stub"] = _make_sdk_stub(
        result["name"],
        result["prototype"],
        result.get("decompiled"),
    )

    if result["prototype"] is None:
        result["notes"].append("Prototype is missing or weak; verify calling convention before porting.")
    if not result["stack_frame"]:
        result["notes"].append("No stack-frame variables were recovered; expect manual local-variable cleanup.")
    if not result["strings"] and not result["constants"]:
        result["notes"].append("Function has little literal signal; validate logic carefully against assembly.")

    return result
