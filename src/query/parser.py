import re

_PLIKE = re.compile(
    r'PATTERN_LIKE\(\s*(\w+)\s*,\s*(:?\w+)\s*'
    r'(?:,\s*threshold\s*=>\s*([\d.]+))?'
    r'(?:,\s*recall_target\s*=>\s*([\d.]+))?'
    r'(?:,\s*version\s*=>\s*(\d+))?\s*\)', re.IGNORECASE)
_LIMIT   = re.compile(r'\bLIMIT\s+(\d+)\b',             re.IGNORECASE)
_WHERE   = re.compile(r'\bWHERE\b(.+?)(?:\bORDER\b|\bLIMIT\b|$)',
                       re.IGNORECASE | re.DOTALL)

def parse(sql):
    plan = {"table": None, "pattern_like": None, "rel_predicates": [],
            "rel_where": "", "order_by": None, "limit": 20}

    m = re.search(r'\bFROM\s+(\w+)\b', sql, re.IGNORECASE)
    if m: plan["table"] = m.group(1)

    m = _LIMIT.search(sql)
    if m: plan["limit"] = int(m.group(1))

    m = re.search(r'\bORDER\s+BY\s+(\w+)\s*(ASC|DESC)?', sql, re.IGNORECASE)
    if m: plan["order_by"] = {"col": m.group(1), "dir": (m.group(2) or "DESC").upper()}

    wm = _WHERE.search(sql)
    where = wm.group(1).strip() if wm else ""

    pm = _PLIKE.search(where)
    if pm:
        plan["pattern_like"] = {
            "col":          pm.group(1),
            "param":        pm.group(2),
            "threshold":    float(pm.group(3)) if pm.group(3) else 0.85,
            "recall_target": float(pm.group(4)) if pm.group(4) else 0.95,
            "version":      int(pm.group(5)) if pm.group(5) else None,
        }
        rel = _PLIKE.sub("", where)
        rel = re.sub(r'^\s*AND\s*|\s*AND\s*$', "", rel.strip(), flags=re.IGNORECASE).strip()
        plan["rel_where"] = rel
        plan["rel_predicates"] = _parse_preds(rel)
    else:
        plan["rel_where"] = where
        plan["rel_predicates"] = _parse_preds(where)
    return plan

def _parse_preds(where):
    preds = []
    for part in re.split(r'\bAND\b', where, flags=re.IGNORECASE):
        part = part.strip()
        if not part: continue
        for op in [">=","<=","<>","!=",">","<","=",r"\bBETWEEN\b",r"\bIN\b"]:
            if re.search(op, part, re.IGNORECASE):
                preds.append({"col": part.split()[0], "op": op.strip("\\b"), "raw": part})
                break
    return preds
