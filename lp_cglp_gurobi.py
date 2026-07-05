#!/usr/bin/env python3
"""
lp_cglp_gurobi.py - lift-and-project cuts for the Steiner bidirected (BC) model, using Gurobi.

Idea in one line: take an edge whose LP value y is a fraction (between 0 and 1). In any real
solution that edge is either not used (y=0) or used (y=1), so we build a small helper LP that
produces one cut which removes the current fractional point but keeps every 0/1 solution. Because
of that, the cut can never cut off a real solution -- it is valid by construction, so we never need
the optimum and the bound can never go above it. We add these cuts round by round to raise the LP
bound. The solver only solves the LPs we give it; we do not use any of its built-in cuts.

The --ip flag solves the integer model at the END only, to report how much of the gap we closed and
to double-check the bound never went above the optimum.

Quick check:  python lp_cglp_gurobi.py data/i080-235.stp --ip   should print gap closed 100% and VALIDITY OK.

Requires: gurobipy, scipy, numpy, core.py
"""
import argparse, time
import numpy as np
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
from core import read_stp, Instance


def build_bc(inst, relax=True):
    """Builds the Steiner bidirected (BC) model: design variables y (one per edge), flow f for each
       commodity, and the forward/reverse helpers zf, zr. relax=True makes y continuous (the LP);
       relax=False makes y 0/1 (the integer model). Variable order is y, f, zf, zr."""
    a, A, K, root = inst.a, inst.A, inst.K, inst.root
    m = gp.Model(); m.Params.OutputFlag = 0
    ydt = GRB.BINARY if not relax else GRB.CONTINUOUS
    y = m.addVars(a, lb=0.0, ub=1.0, vtype=ydt, name="y")
    f = m.addVars(K, A, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="f")
    zf = m.addVars(a, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="zf")
    zr = m.addVars(a, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="zr")
    m.setObjective(gp.quicksum(inst.c1[e] * y[e] for e in range(a)), GRB.MINIMIZE)
    terms = inst.terminals
    n_flow = 0
    # flow balance: each commodity sends one unit from the root to its terminal
    for k in range(K):
        tk = terms[k + 1]
        for v in range(1, inst.n + 1):
            expr = (gp.quicksum(f[k, arc] for arc in inst.out_arcs[v]) -
                    gp.quicksum(f[k, arc] for arc in inst.in_arcs[v]))
            rhs = 1.0 if v == root else (-1.0 if v == tk else 0.0)
            m.addConstr(expr == rhs); n_flow += 1
    n_couple = 0
    # flow can only use an edge that is bought: flow <= zf/zr, and zf + zr <= y
    for e in range(a):
        for k in range(K):
            m.addConstr(f[k, e] - zf[e] <= 0.0); n_couple += 1
            m.addConstr(f[k, e + a] - zr[e] <= 0.0); n_couple += 1
        m.addConstr(zf[e] + zr[e] - y[e] <= 0.0); n_couple += 1
    m.update()
    meta = dict(a=a, A=A, K=K, nY=a, nF=K * A, nv=a + K * A + 2 * a,
                n_flow=n_flow, n_couple=n_couple)
    return m, meta


def ge_system(m):
    """Rewrites every constraint and every variable bound of the model in one common form
       'row . x >= number'. The cut helper below needs all the constraints in this single form.
       Returns the stacked rows, the right-hand-side numbers, and the number of variables."""
    A = m.getA().tocsr()                       # num_constr x num_var (scipy sparse)
    senses = np.array([c.Sense for c in m.getConstrs()])
    rhs = np.array([c.RHS for c in m.getConstrs()])
    vars_ = m.getVars()
    n = len(vars_)
    lb = np.array([v.LB for v in vars_]); ub = np.array([v.UB for v in vars_])
    rows = []; b = []
    for i in range(A.shape[0]):
        ai = A.getrow(i); s = senses[i]
        if s == '>':
            rows.append(ai); b.append(rhs[i])                  # a.x >= rhs
        elif s == '<':
            rows.append(-ai); b.append(-rhs[i])                # -a.x >= -rhs
        else:  # '='  -> two inequalities
            rows.append(ai); b.append(rhs[i])
            rows.append(-ai); b.append(-rhs[i])
    I = sp.eye(n, format='csr')
    for j in range(n):
        if lb[j] > -GRB.INFINITY + 1:
            rows.append(I.getrow(j)); b.append(lb[j])          # x_j >= lb
    for j in range(n):
        if ub[j] < GRB.INFINITY - 1:
            rows.append(-I.getrow(j)); b.append(-ub[j])        # -x_j >= -ub
    Age = sp.vstack(rows, format='csr')
    return Age, np.array(b), n


def lp_cut(Age, bge, n, xstar, p, At=None):
    """The cut helper. For the fractional edge p, it builds and solves a small LP that produces one
       lift-and-project cut 'alpha . x >= beta'. The cut is built so that it holds for every real
       solution where edge p is either not used (x_p <= 0) or used (x_p >= 1), which is every 0/1
       solution -- so the cut is always valid, it just removes the current fractional point.
       Returns (alpha, beta, how much the current point breaks the cut) or None.
       (The helper's own variables u, v, u0, v0 combine the model's constraints into the cut; this
       is the standard lift-and-project construction.)"""
    M = Age.shape[0]
    if At is None:
        At = Age.transpose().tocsc()           # n x M
    res = bge - Age @ xstar                     # length M
    ep = sp.csc_matrix((np.array([1.0]), (np.array([p]), np.array([0]))), shape=(n, 1))
    # Bmat (n x 2M+2): [At | -At | ep | ep]
    Bmat = sp.hstack([At, -At, ep, ep], format='csr')
    g = gp.Model(); g.Params.OutputFlag = 0
    w = g.addMVar(2 * M + 2, lb=0.0)
    g.addConstr(Bmat @ w == np.zeros(n))                       # makes the cut consistent on both sides
    g.addConstr(bge @ w[:M] - bge @ w[M:2 * M] + w[2 * M] == 0.0)   # ties the right-hand side together
    g.addConstr(w.sum() == 1.0)                                # scale the weights so the LP is bounded
    c = np.zeros(2 * M + 2); c[:M] = res; c[2 * M] = (1.0 - xstar[p])
    g.setObjective(c @ w, GRB.MAXIMIZE)
    g.optimize()
    if g.Status != GRB.OPTIMAL:
        return None
    wv = w.X
    u = wv[:M]; u0 = wv[2 * M]
    alpha = (Age.transpose() @ u); alpha[p] += u0
    beta = float(bge @ u + u0)
    viol = beta - float(alpha @ xstar)
    return alpha, beta, viol


def loop(stp, rounds=30, per_round=15, budget=1e9, do_ip=False, ip_time=200.0, ip_only=False):
    edges, terms = read_stp(stp); inst = Instance(edges, terms)
    if ip_only:                                   # ONLY measure the Gurobi IP solve time, then exit
        mi, _ = build_bc(inst, relax=False)
        if ip_time and ip_time > 0:
            mi.Params.TimeLimit = ip_time
        mi.optimize()
        proven = (mi.Status == GRB.OPTIMAL)
        val = mi.ObjVal if mi.SolCount > 0 else float('nan')
        print("=" * 60)
        print(f"instance     : {stp}")
        print(f"  nodes      : {inst.n}   edges: {inst.a}   commodities K: {inst.K}")
        print(f"  IP optimum : {val:.3f}   "
              f"[{'optimal, proven' if proven else f'NOT proven optimal (status {mi.Status})'}]")
        print(f"  Gurobi IP solve time : {mi.Runtime:.2f} s")
        print("=" * 60)
        return
    m, meta = build_bc(inst, relax=True)
    m.optimize()
    lp0 = m.ObjVal
    print("=" * 60)
    print(f"instance     : {stp}")
    print(f"  nodes      : {inst.n}   edges: {inst.a}   commodities K: {inst.K}")
    print(f"  variables  : {meta['nv']} (design y {meta['nY']})   model: BC   solver: Gurobi")
    print(f"LP(BC) bound : {lp0:.3f}")
    Age, bge, n = ge_system(m)
    At = Age.transpose().tocsc()
    print(f"  constraint rows used by the cut helper = {Age.shape[0]}")
    print("-" * 60)
    ipval = None
    if do_ip:
        mi, _ = build_bc(inst, relax=False)
        if ip_time and ip_time > 0:
            mi.Params.TimeLimit = ip_time
        mi.optimize()
        ipval = mi.ObjVal if mi.SolCount > 0 else mi.ObjBound
        print(f"  IP optimum : {ipval:.3f}   (Gurobi IP solve time {mi.Runtime:.2f}s)")

    x = m.getVars()
    t0 = time.time(); bound = lp0; best = lp0; total_cuts = 0
    cuts = []           # cuts we have added so far (kept so we can drop the useless ones later)
    DROP_AFTER = 3      # drop a cut if it has been loose for this many rounds in a row
    for rd in range(rounds):
        if time.time() - t0 > budget:
            print("  (budget)"); break
        xstar = np.array(m.getAttr('X', x))
        # pick the edges whose LP value is a fraction (not 0 and not 1) -- these are the ones we can cut
        fr = [(abs(xstar[e] - round(xstar[e])), e) for e in range(meta['a'])
              if 1e-4 < xstar[e] - np.floor(xstar[e]) < 1 - 1e-4]
        fr.sort(reverse=True)
        added = 0
        new_idx = len(cuts)
        for _, p in fr[:per_round]:
            if time.time() - t0 > budget:        # stop mid-round once the time limit is hit
                break
            out = lp_cut(Age, bge, n, xstar, p, At=At)
            if out is None:
                continue
            alpha, beta, viol = out
            if viol > 1e-5:                      # the cut removes the current point -> add it
                idx = np.nonzero(np.abs(alpha) > 1e-9)[0]
                con = m.addConstr(gp.quicksum(alpha[j] * x[int(j)] for j in idx) >= beta)
                cuts.append(dict(con=con, idx=idx, val=alpha[idx], beta=beta, streak=0))
                added += 1
        if added == 0:
            print(f"  round {rd}: no L&P cut"); break
        m.optimize(); nb = m.ObjVal
        best = max(best, nb)
        # tidy-up: remove cuts that have stayed loose (not tight) for a few rounds, to keep the LP small
        xstar2 = np.array(m.getAttr('X', x))
        keep = []; dropped = 0
        for ci, cd in enumerate(cuts):
            slack = float(cd['val'] @ xstar2[cd['idx']]) - cd['beta']
            cd['streak'] = cd['streak'] + 1 if slack > 1e-6 else 0
            if cd['streak'] >= DROP_AFTER and ci < new_idx:
                m.remove(cd['con']); dropped += 1
            else:
                keep.append(cd)
        if dropped:
            cuts = keep; m.optimize()
        tag = f" (-{dropped} loose)" if dropped else ""
        print(f"  round {rd}: +{added} L&P cuts{tag} -> bound {nb:.4f} "
              f"(best {best:.4f}, +{best - lp0:.3f})  [{time.time() - t0:.0f}s]")
        bound = nb; total_cuts += added
    bound = best
    print("-" * 60)
    print(f"FINAL L&P bound: {bound:.4f}   lift {bound - lp0:+.3f}   ({total_cuts} L&P cuts generated)")
    if ipval is not None:
        closed = (bound - lp0) / (ipval - lp0) * 100 if ipval > lp0 + 1e-9 else 100.0
        print(f"IP optimum   : {ipval:.3f}   gap closed: {closed:.1f}%   "
              f"VALIDITY: {'OK' if bound <= ipval + 1e-4 else 'VIOLATED!!'}")
    total_time = time.time() - t0
    print(f"Total computation time : {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Lift-and-project cuts for the Steiner BC model (Gurobi)")
    ap.add_argument("instance")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--per-round", type=int, default=15, dest="per_round")
    ap.add_argument("--budget", type=float, default=1e9,
                    help="wall-clock seconds for the cut loop (e.g. 7200 to match the professor)")
    ap.add_argument("--ip", action="store_true", help="solve IP at end for closure + validity")
    ap.add_argument("--ip-only", action="store_true", dest="ip_only",
                    help="ONLY solve the Gurobi IP and report its solve time, then exit (no L&P)")
    ap.add_argument("--ip-time", type=float, default=200.0,
                    help="IP time limit (s); <=0 means no limit (solve to optimality)")
    a = ap.parse_args()
    loop(a.instance, rounds=a.rounds, per_round=a.per_round, budget=a.budget,
         do_ip=a.ip, ip_time=a.ip_time, ip_only=a.ip_only)