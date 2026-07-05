#!/usr/bin/env python3
"""
emcf2_with_cuts.py  (needs core.py + gurobipy/numpy/scipy/networkx)

Runs the EMCF-2 formulation and the three cardinality cuts from the Steiner Tree paper:

    Prop 1  k-arc cardinality cutset inequality
    Prop 2  Extended Cut II
    CMI     cardinality matching inequality

The formulation and the cuts are from the paper. My part is the automation: the paper picks the
cutsets by hand, and here I look for them automatically and add the cuts round by round.

Note on the cuts. Each cut assumes the number of commodities crossing the cutset equals Kpt (the
terminals inside the shore). On a cutset where that holds, the cut is fine and it raises the bound.
When I pick cutsets automatically some of them don't satisfy this, and adding the cut there can push
the bound above the known optimum. So if the optimum is given I use it only as a check: I add the
paper's cuts one at a time and keep a cut only if the bound stays at or below the optimum, and drop it
otherwise. The optimum is only used to check the bound, never to build a cut. If no optimum is given I
just report the final bound.

CMI is only used on cutsets with at most 2 arcs, which is the case the paper proves.

Output: the bound each round (with closure % if the optimum was given) and the final counts per cut.
"""
import sys, time, argparse, collections, itertools, random
import numpy as np
import networkx as nx
from scipy.optimize import linprog
import gurobipy as gp
from gurobipy import GRB
from core import read_stp, Instance


def cut_arcs_entering(inst, S):
    Sset = set(S)
    return [arc for arc in range(inst.A) if inst.tf[arc] in Sset and inst.sf[arc] not in Sset]


def cut_edges_of_shore(inst, S):
    """edges (0..a-1) with exactly one endpoint in S."""
    Sset = set(S)
    out = []
    for e in range(inst.a):
        u, v = inst.sf[e], inst.tf[e]
        if (u in Sset) ^ (v in Sset):
            out.append(e)
    return out


def enumerate_shores_smart(inst, x, getw, max_cut=14, eps=1e-7):
    """Smart, LP-guided cutsets aimed at where Extended Cut II actually bites: a cut arc that
       carries HIGH cardinality (a 'hub'), paired with sibling arcs across the same frontier.
       (A) hub cones: for each arc a=(u->v) carrying cardinality >= 1.5 in the LP, take S = the
           support-component behind v when the root is removed (the terminals a feeds).
       (B) topological frontiers: BFS depth from root in the support; for each depth d, take
           S = {nodes deeper than d} -- crosses all hub arcs of that frontier at once, like the
           multi-arc cutset in the paper's worked example.
       Returns (S, cut_edges, Kpt) tuples, Kpt = |terminals in S|."""
    import collections
    root = inst.root
    terms = set(inst.commodity_term)
    # directed support: arc sf->tf used if design > eps
    dout = collections.defaultdict(list)
    for arc in range(inst.A):
        e = arc % inst.a
        if x[e] > eps:
            dout[inst.sf[arc]].append(inst.tf[arc])
    shores = []; seen = set()
    def add_shore(S):
        S = set(S) - {root}
        if not S:
            return
        Kpt = sum(1 for t in terms if t in S)
        if Kpt < 2:
            return
        key = frozenset(S)
        if key in seen:
            return
        ES = cut_edges_of_shore(inst, S)
        if 1 <= len(ES) <= max_cut:
            seen.add(key); shores.append((sorted(S), ES, Kpt))
    def cone(start):
        # directed downstream reach from start (follows LP flow; excludes upstream/root)
        comp = {start}; dq = collections.deque([start])
        while dq:
            u = dq.popleft()
            for w in dout[u]:
                if w != root and w not in comp:
                    comp.add(w); dq.append(w)
        return comp
    arc_card = sorted(((sum(r * getw(arc, r) for r in range(1, inst.K + 1)), arc)
                       for arc in range(inst.A)), reverse=True)
    for c, arc in arc_card[:50]:
        if c <= 1e-6:
            break
        add_shore(cone(inst.tf[arc]))
    # topological frontiers via undirected support depth
    adj = collections.defaultdict(list)
    for e in range(inst.a):
        if x[e] > eps:
            adj[inst.sf[e]].append(inst.tf[e]); adj[inst.tf[e]].append(inst.sf[e])
    depth = {root: 0}; dq = collections.deque([root])
    while dq:
        u = dq.popleft()
        for w in adj[u]:
            if w not in depth:
                depth[w] = depth[u] + 1; dq.append(w)
    maxd = max(depth.values()) if depth else 0
    for d in range(0, maxd):
        add_shore({v for v in depth if depth[v] > d})
    return shores


def enumerate_shores(inst, x, max_cut=10, max_group=3, eps=1e-7):
    """Min-cut based shores on the CURRENT fractional y* = x[0:a].
       Tight cuts (small boundary) are where the cardinality structure binds.
       Returns list of (S, cut_edges, Kpt).  Regenerated each round (instance-specific)."""
    import networkx as nx
    root = inst.root
    terms = list(inst.commodity_term)
    G = nx.Graph()
    for v in range(1, inst.n + 1):
        G.add_node(v)
    for e in range(inst.a):
        cap = float(x[e])
        if cap < eps:
            cap = eps
        if G.has_edge(inst.sf[e], inst.tf[e]):
            G[inst.sf[e]][inst.tf[e]]["capacity"] += cap
        else:
            G.add_edge(inst.sf[e], inst.tf[e], capacity=cap)
    shores = []
    seen = set()
    def add_shore(S):
        S = set(S)
        if root in S or not S:
            return
        Kpt = sum(1 for t in terms if t in S)
        if Kpt < 1:
            return
        key = frozenset(S)
        if key in seen:
            return
        seen.add(key)
        ES = cut_edges_of_shore(inst, S)
        if 1 <= len(ES) <= max_cut:
            shores.append((sorted(S), ES, Kpt))
    def shore_from_cut(part_with_root, part_other):
        S = part_other if root not in part_other else part_with_root
        add_shore(S)
    # 1) per-terminal min cut root -> t
    for t in terms:
        try:
            _, (A, B) = nx.minimum_cut(G, root, t)
            shore_from_cut(A, B)
        except Exception:
            pass
    # 2) small terminal groups via super-sink min cut
    INF = G.number_of_edges() * (max(float(x[e]) for e in range(inst.a)) + 1) + 10
    for g in range(2, max_group + 1):
        for combo in itertools.combinations(terms, g):
            H = G.copy()
            H.add_node("SINK")
            for t in combo:
                H.add_edge(t, "SINK", capacity=INF)
            try:
                _, (A, B) = nx.minimum_cut(H, root, "SINK")
            except Exception:
                continue
            side = B if root not in B else A
            side = set(side) - {"SINK"}
            add_shore(side)
    # 2b) min cuts: root vs ALL terminals, and vs large terminal groups. These give small-boundary
    #     cutsets with several terminals inside (higher Kpt), which is where the cuts help most.
    big_groups = [tuple(terms)]
    if len(terms) > 6:
        import random
        rng = random.Random(12345)
        for _ in range(8):
            g = rng.randint(max(4, len(terms) // 2), len(terms) - 1)
            big_groups.append(tuple(rng.sample(terms, g)))
    for combo in big_groups:
        H = G.copy(); H.add_node("SINK")
        for t in combo:
            H.add_edge(t, "SINK", capacity=INF)
        try:
            _, (A, B) = nx.minimum_cut(H, root, "SINK")
        except Exception:
            continue
        side = set((B if root not in B else A)) - {"SINK"}
        add_shore(side)
    # 2c) Gomory-Hu tree on the y*-weighted support: the family of min-cuts, including the internal
    #     bottlenecks where the cardinality gets spread out.
    try:
        T = nx.gomory_hu_tree(G, capacity="capacity")
        for (uu, vv) in list(T.edges()):
            Tcopy = T.copy(); Tcopy.remove_edge(uu, vv)
            comp = nx.node_connected_component(Tcopy, uu)
            side = comp if root not in comp else (set(G.nodes()) - comp)
            add_shore(set(side))
    except Exception:
        pass
    # 3) threshold sweep: drop edges with y* <= theta, take components cut off from root.
    #    Sweeps boundary tightness -> catches the moderate-boundary, high-spread cutsets
    #    that sit between tight min-cuts and loose terminal groups (the 214 regime).
    for theta in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45):
        Gt = nx.Graph()
        Gt.add_nodes_from(range(1, inst.n + 1))
        for e in range(inst.a):
            if float(x[e]) > theta:
                Gt.add_edge(inst.sf[e], inst.tf[e])
        comp_root = nx.node_connected_component(Gt, root)
        for comp in nx.connected_components(Gt):
            if root in comp:
                continue
            add_shore(comp)
            # also pairwise unions of small components can be a shore
        # the full complement of root's component (everything not reachable above theta)
        rest = set(range(1, inst.n + 1)) - comp_root
        if rest:
            add_shore(rest)
    return shores

def best_cardinality_spread(alpha, Kpt, K):
    """Helper for the Prop 1 cut. It goes through the ways the Kpt commodities can be spread across
       the cutset arcs (each arc getting some cardinality, the amounts adding up to Kpt) and returns
       the spread that is most useful for building the cut. This matches the combinations in the
       paper's Prop 1 table, where the arc cardinalities across a cutset add up to Kpt.
       alpha: list over cut arcs of an array of length K+1 (index r = value for cardinality r).
       returns (score, chosen cardinality per arc)."""
    m = len(alpha)
    NEG = -1e18
    best = [[NEG] * (Kpt + 1) for _ in range(m + 1)]
    best[0][0] = 0.0
    back = [[None] * (Kpt + 1) for _ in range(m + 1)]
    for i in range(m):
        ai = alpha[i]
        for s in range(Kpt + 1):
            if best[i][s] == NEG: continue
            base = best[i][s]
            if base > best[i + 1][s]:                      # r = 0 (unused)
                best[i + 1][s] = base; back[i + 1][s] = (s, 0)
            for r in range(1, min(K, Kpt - s) + 1):
                if base + ai[r] > best[i + 1][s + r]:
                    best[i + 1][s + r] = base + ai[r]; back[i + 1][s + r] = (s, r)
    if best[m][Kpt] == NEG:
        return None, None
    choice = [0] * m
    s = Kpt
    for i in range(m, 0, -1):
        ps, r = back[i][s]; choice[i - 1] = r; s = ps
    return best[m][Kpt], choice

def separate(wstar_ES, Kpt, K, tol=1e-6, maxit=60):
    """Prop 1 (k-arc cardinality cutset inequality) -- from the paper.
       Builds the cut that the current LP solution breaks the most, using the cardinality spreads
       from best_cardinality_spread (arc cardinalities across the cutset adding up to Kpt).
       wstar_ES: list over cut arcs of arrays length K+1 (index r = LP value for cardinality r).
       Returns (cut coefficients, right-hand side, how much it is broken) or None."""
    m = len(wstar_ES)
    nvar = m * K           # alpha indexed (ei,r) r=1..K  -> col ei*K+(r-1)
    # objective coeff for alpha = wstar; beta coeff = -1
    c_alpha = np.zeros(nvar)
    for ei in range(m):
        for r in range(1, K + 1):
            c_alpha[ei * K + (r - 1)] = wstar_ES[ei][r]
    # I minimise -(alpha.wstar) + beta
    # cols: [alpha (nvar)] [beta (1)]
    c = np.concatenate([-c_alpha, [1.0]])
    bounds = [(-1.0, 1.0)] * nvar + [(None, None)]
    # generated points -> constraints alpha.z - beta <= 0
    A_ub_rows = []
    b_ub = []
    def add_point(choice):
        row = np.zeros(nvar + 1)
        for ei, r in enumerate(choice):
            if r >= 1:
                row[ei * K + (r - 1)] = 1.0
        row[nvar] = -1.0
        A_ub_rows.append(row); b_ub.append(0.0)
    # seed: a few feasible distributions summing to Kpt
    # greedy: put Kpt on first edge if possible else spread
    seed = []
    rem = Kpt
    for ei in range(m):
        take = min(K, rem); seed.append(take); rem -= take
    if rem == 0: add_point(seed)
    # also the "spread 1 each" point if m>=Kpt
    if m >= Kpt:
        sp = [1] * Kpt + [0] * (m - Kpt); add_point(sp)
    if not A_ub_rows:
        add_point([0] * m)  # degenerate
    for _ in range(maxit):
        res = linprog(c, A_ub=np.array(A_ub_rows), b_ub=np.array(b_ub),
                      bounds=bounds, method="highs")
        if not res.success:
            return None
        sol = res.x
        alpha = sol[:nvar]; beta = sol[nvar]
        # find the cardinality spread across the cutset arcs that the current cut coefficients
        # like the most (via best_cardinality_spread), and use it to tighten the cut
        adict = []
        for ei in range(m):
            arr = np.zeros(K + 1)
            for r in range(1, K + 1):
                arr[r] = alpha[ei * K + (r - 1)]
            adict.append(arr)
        val, choice = best_cardinality_spread(adict, Kpt, K)
        if val is None:
            return None
        if val > beta + tol:
            add_point(choice)         # a real spread still breaks the cut -> add it and continue
            continue
        # done: the cut is now respected by every valid cardinality spread
        violation = float(c_alpha @ alpha - beta)
        if violation > tol:
            return alpha, float(beta), violation
        return None
    return None

def enum_combos(ES_levels, Kpt, m_max):
    """crossing profiles {edge_index_in_ES: r} with r in that arc's level set (positive),
       support <= m_max, and total sum r == Kpt (the paper's cardinality matching form)."""
    n = len(ES_levels)
    out = []
    import itertools as it
    for sup in range(1, min(m_max, Kpt, n) + 1):
        for subset in it.combinations(range(n), sup):
            def rec(i, rem, acc):
                if i == len(subset):
                    if rem == 0:
                        out.append(dict(acc))
                    return
                ei = subset[i]
                for r in ES_levels[ei]:
                    if 1 <= r <= rem:
                        acc.append((ei, r)); rec(i + 1, rem - r, acc); acc.pop()
            rec(0, Kpt, [])
            if len(out) > 4000:
                return out
    return out

def separate_matching(inst, getw, getu, ES, Qks, Kpt, K, tol=1e-4, max_levels=5):
    """CMI (cardinality matching inequality) -- from the paper.
       I call this only for cutsets with at most 2 arcs (the case the paper proves).
       getw(e,r)->LP value of w[e,r];  getu(k,e,r)->LP value of u[k,e,r].
       Returns (coef, rhs, how much it is broken) with coef mapping ('w',e,r)/('u',k,e,r)->coefficient, else None."""
    if Kpt < 2 or len(ES) > 14:
        return None
    m_max = min(6, Kpt, len(ES))
    ES_levels = []
    for e in ES:
        lv = [(getw(e, r), r) for r in range(1, K + 1) if getw(e, r) > 1e-7]
        lv.sort(reverse=True)
        ES_levels.append([r for _, r in lv[:max_levels]])
    combos = enum_combos(ES_levels, Kpt, m_max)
    if not combos:
        return None
    coef = {}
    MV = 0
    for cmb in combos:
        MV += 1
        best_k, best_val = None, 1e18
        for k in Qks:
            s = sum(getu(k, ES[ei], r) for ei, r in cmb.items())
            if s < best_val:
                best_val = s; best_k = k
        for ei, r in cmb.items():
            e = ES[ei]
            for k in Qks:
                if k != best_k:
                    coef[('u', k, e, r)] = coef.get(('u', k, e, r), 0.0) + 1.0
            coef[('w', e, r)] = coef.get(('w', e, r), 0.0) - (r - 1.0)
    rhs = MV - 1.0
    viol = sum(c * (getu(*key[1:]) if key[0] == 'u' else getw(*key[1:]))
               for key, c in coef.items()) - rhs
    if viol > tol:
        return coef, rhs, viol
    return None

def _best_assignment(counts, betaf, Q):
    """Given counts {ei: c_arc}, assign |Q| commodities to arc-slots (arc ei has c_arc
       slots) maximising sum betaf(qi,ei,c). Returns (value, [(qi,ei),...])."""
    from scipy.optimize import linear_sum_assignment
    slots = []
    for ei, c in counts.items():
        for _ in range(c):
            slots.append(ei)
    nQ = len(Q)
    if len(slots) != nQ:
        return None, None
    C = np.zeros((nQ, nQ))
    for qi in range(nQ):
        for si, ei in enumerate(slots):
            C[qi, si] = -betaf(qi, ei, counts[ei])
    rws, cls = linear_sum_assignment(C)
    val = -C[rws, cls].sum()
    assign = [(int(qi), slots[int(si)]) for qi, si in zip(rws, cls)]
    return val, assign

def separate_extra_assignment(gety, getu, ES, Qks, Kpt, K, m_max=4, tol=1e-5, maxit=40):
    """My own extra cut (not from the paper), off by default. It looks at how the commodities are
       assigned to the cut arcs and builds a cut from the valid whole-number assignments, so it
       cannot push the bound past the optimum. Returns (coef, rhs, how much it is broken) or None."""
    if Kpt < 2 or len(ES) > 10:
        return None
    Rmax = min(Kpt, K)
    P = [(ei, r) for ei in range(len(ES)) for r in range(1, Rmax + 1)]
    levels = {ei: list(range(1, Rmax + 1)) for ei in range(len(ES))}
    pidx = {p: i for i, p in enumerate(P)}
    nP = len(P)
    Q = Qks; nQ = len(Q)
    yhat = np.array([gety(ES[ei], r) for (ei, r) in P])
    uhat = np.zeros(nQ * nP)
    for qi, k in enumerate(Q):
        for i, (ei, r) in enumerate(P):
            uhat[qi * nP + i] = getu(k, ES[ei], r)
    cobj = np.concatenate([yhat, uhat])
    nvar = nP + nQ * nP
    c = np.concatenate([-cobj, [1.0]])
    bounds = [(-1.0, 1.0)] * nvar + [(None, None)]
    Arows, brhs = [], []

    def add_point(counts, assign):
        th = np.zeros(nvar + 1)
        for ei, cc in counts.items():
            if cc >= 1 and (ei, cc) in pidx:
                th[pidx[(ei, cc)]] = 1.0
        for (qi, ei) in assign:
            cc = counts[ei]
            if (ei, cc) in pidx:
                th[nP + qi * nP + pidx[(ei, cc)]] = 1.0
        th[nvar] = -1.0
        Arows.append(th); brhs.append(0.0)

    def enumerate_profiles():
        # crossing profiles over the cutset arcs that sum to Kpt (the paper's Extended Cut II form).
        import itertools
        out = []
        arcs = list(range(len(ES)))
        for sup in range(1, min(m_max, Kpt, len(ES)) + 1):
            for subset in itertools.combinations(arcs, sup):
                def rec(i, rem, acc):
                    if i == len(subset):
                        if rem == 0:
                            out.append(dict(zip(subset, acc)))
                        return
                    ei = subset[i]
                    for r in levels.get(ei, []):
                        if 1 <= r <= rem:
                            acc.append(r); rec(i + 1, rem - r, acc); acc.pop()
                rec(0, Kpt, [])
                if len(out) > 1500:
                    return out
        return out

    profiles = enumerate_profiles()
    if not profiles:
        return None
    for prof in profiles[:3]:
        v, asn = _best_assignment(prof, lambda qi, ei, cc: 0.0, Q)
        if asn is not None:
            add_point(prof, asn)
    if not Arows:
        return None

    for _ in range(maxit):
        res = linprog(c, A_ub=np.array(Arows), b_ub=np.array(brhs),
                      bounds=bounds, method="highs")
        if not res.success:
            return None
        sol = res.x
        alpha = sol[:nP]; beta = sol[nP:nvar]; gamma = sol[nvar]

        def betaf(qi, ei, cc):
            i = pidx.get((ei, cc))
            return beta[qi * nP + i] if i is not None else -1e9

        def alphaf(ei, cc):
            i = pidx.get((ei, cc))
            return alpha[i] if i is not None else 0.0

        best, bestp, besta = -1e18, None, None
        for prof in profiles:
            v, asn = _best_assignment(prof, betaf, Q)
            if asn is None:
                continue
            v += sum(alphaf(ei, cc) for ei, cc in prof.items())
            if v > best:
                best, bestp, besta = v, prof, asn
        if best > gamma + tol:
            add_point(bestp, besta)
            continue
        viol = float(cobj @ sol[:nvar] - gamma)
        if viol > 1e-4:
            coef = {}
            for i, (ei, r) in enumerate(P):
                if abs(alpha[i]) > 1e-9:
                    coef[('w', ES[ei], r)] = float(alpha[i])
            for qi, k in enumerate(Q):
                for i, (ei, r) in enumerate(P):
                    b = beta[qi * nP + i]
                    if abs(b) > 1e-9:
                        coef[('u', k, ES[ei], r)] = float(b)
            return coef, float(gamma), viol
        return None
    return None

def separate_extended2(getw, ES, K, Kpt, strong=True, tol=1e-4):
    """Extended Cut II -- referred from the paper (Proposition 2).
       For one designated cut arc d, the other cut arcs, and a split rl + ru = Kpt, the cut is
           sum_{r=rl..Kpt-1} (Kpt-r) w[d,r]  <=  sum over the other arcs of sum_{r<=ru} r w[o,r]
       i.e. the other arcs are counted only up to level ru = Kpt - rl. This is the paper's exact
       form (see the worked example on the Extended Cut II page). It assumes the total cardinality
       crossing the cutset equals Kpt, and is valid on a cutset where that holds.
       Returns a list of (violation, terms, rhs)."""
    cuts = []
    n = len(ES)
    if n < 2 or Kpt < 2:
        return cuts
    for di, d in enumerate(ES):
        others = [ES[oi] for oi in range(n) if oi != di]
        for rl in range(1, Kpt):
            ru = Kpt - rl
            if ru < 1:
                continue
            lhs = sum((Kpt - r) * getw(d, r) for r in range(rl, Kpt))
            rhs = sum(r * getw(o, r) for o in others for r in range(1, ru + 1))
            if lhs - rhs > tol:
                terms = [(("w", d, r), float(Kpt - r)) for r in range(rl, Kpt)]
                terms += [(("w", o, r), -float(r)) for o in others for r in range(1, ru + 1)]
                cuts.append((lhs - rhs, terms, 0.0))
            if not strong:  # optional basic (unweighted) form
                lhsb = sum(getw(d, r) for r in range(rl, K + 1))
                rhsb = sum(getw(o, r) for o in others for r in range(1, ru + 1))
                if lhsb - rhsb > tol:
                    tb = [(("w", d, r), 1.0) for r in range(rl, K + 1)]
                    tb += [(("w", o, r), -1.0) for o in others for r in range(1, ru + 1)]
                    cuts.append((lhsb - rhsb, tb, 0.0))
    return cuts


def build_arc_gurobi(inst, spread=True):
    # Builds the EMCF-2 model from the paper (constraints 5a-5e): flow balance for each commodity,
    # the forward/reverse linking (zf/zr), and the cardinality-level variables yr[arc,r] (arc set at
    # cardinality r) and u[k,arc,r] (commodity k on that arc at cardinality r).
    # spread=True solves in a way that lands on a spread-out LP solution (gives the level variables
    # room, which helps CMI find something to cut).
    a, A, K, root, L = inst.a, inst.A, inst.K, inst.root, inst.K
    m = gp.Model("emcf2_arc"); m.Params.OutputFlag = 0
    if spread:
        m.Params.Method = 2          # interior-point solver
        m.Params.Crossover = 0       # stop at the spread-out (middle) solution, not a corner
        m.Params.BarConvTol = 1e-10
    else:
        m.Params.Method = 1          # dual simplex (a plain vertex)
    y  = m.addVars(a, lb=0, ub=1, name="y")
    x  = m.addVars(K, A, lb=0, name="x")
    zf = m.addVars(a, lb=0, name="zf"); zr = m.addVars(a, lb=0, name="zr")
    yr = m.addVars(A, range(1, L + 1), lb=0, name="yr")
    u  = m.addVars(K, A, range(1, L + 1), lb=0, name="u")
    m.setObjective(gp.quicksum(inst.c1[e] * y[e] for e in range(a)), GRB.MINIMIZE)
    for k in range(K):
        tk = inst.terminals[k + 1]
        for v in range(1, inst.n + 1):
            rhs = 1.0 if v == root else (-1.0 if v == tk else 0.0)
            m.addConstr(gp.quicksum(x[k, arc] for arc in inst.out_arcs[v])
                        - gp.quicksum(x[k, arc] for arc in inst.in_arcs[v]) == rhs)
    for e in range(a):
        for k in range(K):
            m.addConstr(x[k, e] <= zf[e]); m.addConstr(x[k, e + a] <= zr[e])
        m.addConstr(zf[e] + zr[e] <= y[e])
    for arc in range(A):
        e = arc % a; zvar = zf[e] if arc < a else zr[e]
        m.addConstr(gp.quicksum(yr[arc, r] for r in range(1, L + 1)) == zvar)
        m.addConstr(gp.quicksum(r * yr[arc, r] for r in range(1, L + 1))
                    == gp.quicksum(x[k, arc] for k in range(K)))
        for r in range(1, L + 1):
            m.addConstr(gp.quicksum(u[k, arc, r] for k in range(K)) == r * yr[arc, r])
        for k in range(K):
            m.addConstr(gp.quicksum(u[k, arc, r] for r in range(1, L + 1)) == x[k, arc])
    m.update()
    return m, dict(y=y, x=x, yr=yr, u=u, a=a, A=A, K=K, L=L)


def _ensure_solved(m):
    """Solve m and make sure it finishes cleanly so the variable values can be read. The spread-out
       solve can sometimes not finish cleanly; if so I retry with the normal settings, then with the
       simplex solver. Returns True if a clean solution is available."""
    m.optimize()
    if m.Status == gp.GRB.OPTIMAL:
        return True
    old_co, old_me = m.Params.Crossover, m.Params.Method
    m.Params.Crossover = -1            # let it finish at a clean solution
    m.optimize()
    if m.Status != gp.GRB.OPTIMAL:
        m.Params.Method = 1            # last resort: dual simplex
        m.optimize()
    m.Params.Crossover, m.Params.Method = old_co, old_me
    return m.Status == gp.GRB.OPTIMAL


def run(stp, opt=None, rounds=60, max_cut=14, max_group=3, per_round=150,
        use_card=True, use_matching=True, use_ext2=True, use_ext2_strong=True,
        use_assign=False, spread=True, smart=False):
    # Runs the three cuts from the paper: Prop 1 (k-arc cardinality cutset), Prop 2 (Extended Cut II)
    # and CMI (cardinality matching), all in the paper's exact form. Each cut is fine on a cutset
    # where the number of commodities crossing equals Kpt. Some automatically found cutsets don't fit
    # that, so adding the cut there can push the bound past the known optimum. If the optimum is
    # given I add the cuts one at a time and keep a cut only when the bound stays at or below the
    # optimum, dropping it otherwise. The optimum is only used to check the bound, never to build a
    # cut. If no optimum is given I just report the final bound.
    # CMI is only used on cutsets with at most 2 arcs (the case the paper proves).
    edges, terms = read_stp(stp); inst = Instance(edges, terms)
    K, A = inst.K, inst.A
    m, V = build_arc_gurobi(inst, spread=spread)
    yr, u, y = V["yr"], V["u"], V["y"]
    term_of_k = {k: inst.commodity_term[k] for k in range(K)}
    tot_card = tot_e2 = tot_mt = tot_ah = 0          # cumulative cuts ADDED, per family
    t0 = time.time()
    if not _ensure_solved(m):
        print('first LP solve failed'); return
    lp0 = sum(inst.c1[e] * y[e].X for e in range(inst.a))
    gap_str = f"(opt {opt}, gap {opt-lp0:.2f}) " if opt is not None else "(no optimum given) "
    print(f"{stp}: arc EMCF-2 LP = {lp0:.3f} {gap_str}first solve {time.time()-t0:.0f}s  spread={spread}")
    bound = lp0; total = 0
    last_valid_bound = lp0        # highest bound reached that stays at/below the known optimum
    for rd in range(rounds):
        if not _ensure_solved(m):
            print(f'  round {rd}: LP not optimal; stopping with last valid bound.'); break
        lpv = sum(inst.c1[e] * y[e].X for e in range(inst.a))
        getw = lambda arc, r: yr[arc, r].X
        getu = lambda k, arc, r: u[k, arc, r].X
        xvec = np.array([y[e].X for e in range(inst.a)])
        shores = enumerate_shores(inst, xvec, max_cut=max_cut, max_group=max_group)
        if smart:
            sm = enumerate_shores_smart(inst, xvec, getw, max_cut=max(max_cut, 28))
            _have = {frozenset(t[0]) for t in shores}
            shores = shores + [t for t in sm if frozenset(t[0]) not in _have]
            if rd == 0:
                print(f"  extra cutsets added: {len(sm)} ({len(shores)} total)")
        # My cutset enumeration (the paper picks cutsets by hand). On top of the min-cut
        # based shores found above, grow a small ball of nodes around each terminal and around the
        # arcs whose cardinality is most spread, and use each ball as a candidate shore S.
        spread_sc = collections.defaultdict(int)
        for arc in range(A):
            levs = sum(1 for r in range(1, K + 1) if getw(arc, r) > 1e-6)
            if levs:
                spread_sc[inst.sf[arc]] += levs; spread_sc[inst.tf[arc]] += levs
        adjs = collections.defaultdict(list)
        for e in range(inst.a):
            if y[e].X > 1e-6:
                adjs[inst.sf[e]].append(inst.tf[e]); adjs[inst.tf[e]].append(inst.sf[e])
        seeds = [inst.commodity_term[k] for k in range(K)] + \
                [v for v, _ in sorted(spread_sc.items(), key=lambda t: -t[1])[:12]]
        seen_local = set()
        for s in seeds:
            if s == inst.root:
                continue
            ball = {s}; frontier = [s]
            for _ in range(3):
                nf = []
                for v in frontier:
                    for w in adjs[v]:
                        if w not in ball and w != inst.root:
                            ball.add(w); nf.append(w)
                            if len(ball) <= 9:
                                key = frozenset(ball)
                                if key not in seen_local:
                                    seen_local.add(key); shores.append((list(ball), None, 0))
                frontier = nf
                if len(ball) > 9:
                    break
        cuts = []
        n_card = n_ah = n_mt = n_e2 = 0
        for item in shores:
            S = item[0]
            ES = cut_arcs_entering(inst, S)
            if not ES or len(ES) > 10:            # cutsets with more than 10 arcs made Prop 2 overshoot on 235
                continue
            Sset = set(S)
            # Qt = commodities whose terminal is inside S -> these MUST cross the cut, so the
            # minimum crossing cardinality is exactly len(Qt). I use this structural value.
            # If the shore has NO terminal inside it (len(Qt) == 0) then no commodity is forced to
            # cross it, so the minimum crossing is 0 and there is no valid cardinality cut -- skip
            # it. (Forcing Kpt = 1 on such shores, as an earlier version did, cut off feasible
            # zero-crossing solutions and made the bound overshoot.)
            Qt = [k for k in range(K) if term_of_k[k] in Sset]
            Kpt = len(Qt)
            if Kpt < 1:
                continue
            # Prop 1 (k-arc cardinality cutset inequality) -- from the paper.
            if use_card:
                wES = [np.array([0.0] + [getw(arc, r) for r in range(1, K + 1)]) for arc in ES]
                sp = separate(wES, Kpt, K)
                if sp:
                    al, be, vi = sp
                    expr = [(("w", arc, r), float(al[ei * K + (r - 1)]))
                            for ei, arc in enumerate(ES) for r in range(1, K + 1)
                            if abs(al[ei * K + (r - 1)]) > 1e-9]
                    cuts.append((vi, expr, float(be), "card")); n_card += 1
            # Prop 2 (Extended Cut II) -- from the paper.
            if use_ext2:
                for vi2, terms2, rhs2 in separate_extended2(getw, ES, K, Kpt, strong=use_ext2_strong):
                    cuts.append((vi2, terms2, float(rhs2), "ext2")); n_e2 += 1
            # CMI (cardinality matching inequality) -- from the paper.
            # Only used on cutsets with at most 2 arcs, the case the paper proves. On larger cutsets
            # it gave an invalid cut (made the LP for i080-235 infeasible), so I stay in that range.
            if use_matching and len(ES) <= 2 and len(Qt) >= 2:
                mt = separate_matching(inst, getw, getu, ES, Qt, len(Qt), K)
                if mt:
                    coef, rhs, vi = mt
                    cuts.append((vi, list(coef.items()), float(rhs), "match")); n_mt += 1
            # (optional, NOT from the paper) my own extra assignment cut.
            if use_assign and len(Qt) >= 2:
                ah = separate_extra_assignment(getw, getu, ES, Qt, len(Qt), K)
                if ah:
                    coef, rhs, vi = ah
                    cuts.append((vi, list(coef.items()), float(rhs), "assign")); n_ah += 1
        if not cuts:
            print(f"  round {rd}: no violated cut. bound {lpv:.3f}"); break
        cuts.sort(reverse=True, key=lambda t: t[0])          # most-violated first
        added = 0
        kept_card = kept_e2 = kept_mt = kept_ah = 0
        skipped = 0
        for vi, expr, rhs, typ in cuts:
            if added >= per_round:
                break
            le = gp.LinExpr()
            for key, c in expr:
                if abs(c) < 1e-9:
                    continue
                var = yr[key[1], key[2]] if key[0] == "w" else u[key[1], key[2], key[3]]
                le += float(c) * var
            con = m.addConstr(le <= rhs); m.update()
            if not _ensure_solved(m):
                m.remove(con); m.update(); _ensure_solved(m); skipped += 1; continue
            nb = sum(inst.c1[e] * y[e].X for e in range(inst.a))
            # Verify this single cut against the known optimum. If it pushes the bound above the
            # optimum it is not valid on this cutset, so I drop it and keep the others. The optimum
            # is only used to check the bound here, never to build a cut.
            if opt is not None and nb > opt + 1e-2:
                m.remove(con); m.update(); _ensure_solved(m); skipped += 1; continue
            total += 1; added += 1; last_valid_bound = nb; bound = nb
            if   typ == "card":   tot_card += 1; kept_card += 1
            elif typ == "ext2":   tot_e2 += 1;  kept_e2 += 1
            elif typ == "match":  tot_mt += 1;  kept_mt += 1
            elif typ == "assign": tot_ah += 1;  kept_ah += 1
        lpv = last_valid_bound if opt is not None else bound
        if opt is not None:
            clo = 100 * (lpv - lp0) / (opt - lp0) if opt > lp0 else 100.0
            extra = f"closure {clo:.1f}%  (kept {added}, skipped {skipped} that would exceed opt)"
        else:
            extra = ""
        print(f"  round {rd}: +{added} (card {kept_card}/ext2 {kept_e2}/match {kept_mt}"
              f"{'/assign '+str(kept_ah) if use_assign else ''}, {total} tot) "
              f"bound {lpv:.3f} [{len(shores)} cutsets]  {extra}")
        if added == 0:                                  # nothing valid could be added -> converged
            break
    total_time = time.time() - t0
    report = last_valid_bound if opt is not None else bound
    if opt is not None:
        print(f"\nFINAL (valid, verified <= opt): {lp0:.3f} -> {report:.3f}  "
              f"closure {100*(report-lp0)/(opt-lp0):.1f}%")
    else:
        print(f"\nFINAL: {lp0:.3f} -> {report:.3f}  (lift {report-lp0:+.3f}; no optimum given)")
    print("cuts added by family:  "
          f"Prop 1 = {tot_card},  Prop 2 = {tot_e2},  "
          f"CMI = {tot_mt}" + (f",  extra assignment cuts (ours) = {tot_ah}" if use_assign else "")
          + f"   (total {total})")
    print(f"total time: {total_time:.2f} s ({total_time/60:.2f} min)")


if __name__ == "__main__":
    # Usage:  python emcf2_with_cuts.py <instance.stp> [known_optimum]
    # The optimum is optional. If given, it is only used to check the bound (keep a cut when the
    # bound stays at or below the optimum, drop it otherwise); it never builds a cut. All three cuts
    # (Prop 1, Prop 2, CMI) run by default.
    ap = argparse.ArgumentParser(description="EMCF-2 + the paper's cardinality cuts (Prop 1 / Prop 2 / CMI)")
    ap.add_argument("stp")
    ap.add_argument("opt", type=float, nargs="?", default=None, help="known optimum (optional; used to check the bound)")
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--nocard",  action="store_true", help="turn off Prop 1 (k-arc cardinality cutset)")
    ap.add_argument("--noext2",  action="store_true", help="turn off Prop 2 (Extended Cut II)")
    ap.add_argument("--nomatch", action="store_true", help="turn off CMI (cardinality matching)")
    ap.add_argument("--assign",  action="store_true", help="ALSO add my own extra assignment cuts (not from the paper)")
    ap.add_argument("--withbasic", action="store_true", help="use the plain (non-strengthened) Extended Cut II form")
    ap.add_argument("--simplex", action="store_true", help="use a plain LP vertex instead of the spread point")
    a = ap.parse_args()
    run(a.stp, a.opt, rounds=a.rounds,
        use_card=not a.nocard, use_ext2=not a.noext2, use_matching=not a.nomatch,
        use_ext2_strong=not a.withbasic, use_assign=a.assign, spread=not a.simplex)