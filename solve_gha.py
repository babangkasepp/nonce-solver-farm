#!/usr/bin/env python3
"""NONCE multi-core probSAT solver.
Setup sekali:  pip install numpy numba
Pakai:         python solve.py puzzle.cnf
Hasil: baris 0/1 (1536 karakter) dicetak + disimpan ke answer.txt. Paste isi answer.txt ke chat.
Pakai semua core otomatis; tiap core jalan dengan setting (cb/seed) beda.
"""
import sys, os, time, random, multiprocessing as mp
import numpy as np
from numba import njit

@njit(cache=True)
def run(cl, k, n, occ_ptr, occ, cb, seed, maxflips):
    np.random.seed(seed)
    m = cl.shape[0]
    a = np.random.randint(0, 2, n + 1).astype(np.int8)
    ntrue = np.zeros(m, np.int32); crit = np.zeros(m, np.int32)
    for c in range(m):
        for j in range(k):
            l = cl[c, j]; v = abs(l)
            if (a[v] == 1) == (l > 0):
                ntrue[c] += 1; crit[c] = v
    uns = np.empty(m, np.int32); pos = -np.ones(m, np.int32); nu = 0
    for c in range(m):
        if ntrue[c] == 0:
            uns[nu] = c; pos[c] = nu; nu += 1
    br = np.zeros(n + 1, np.int32)
    for c in range(m):
        if ntrue[c] == 1:
            br[crit[c]] += 1
    tab = np.empty(200, np.float64)
    for i in range(200):
        tab[i] = cb ** (-i)
    probs = np.empty(k, np.float64)
    best = nu
    for f in range(maxflips):
        if nu == 0:
            return a, 0
        c = uns[np.random.randint(nu)]
        s = 0.0
        for j in range(k):
            b = br[abs(cl[c, j])]
            s += tab[b] if b < 200 else 0.0
            probs[j] = s
        r = np.random.random() * s
        j = 0
        while j < k - 1 and probs[j] < r:
            j += 1
        v = abs(cl[c, j]); a[v] = 1 - a[v]
        for t in range(occ_ptr[v], occ_ptr[v + 1]):
            oc = occ[t]; lt = False
            for jj in range(k):
                l = cl[oc, jj]
                if abs(l) == v:
                    lt = (a[v] == 1) == (l > 0); break
            if lt:
                ntrue[oc] += 1
                if ntrue[oc] == 1:
                    p = pos[oc]; last = uns[nu - 1]; uns[p] = last; pos[last] = p; pos[oc] = -1; nu -= 1
                    crit[oc] = v; br[v] += 1
                elif ntrue[oc] == 2:
                    br[crit[oc]] -= 1
            else:
                ntrue[oc] -= 1
                if ntrue[oc] == 0:
                    uns[nu] = oc; pos[oc] = nu; nu += 1; br[v] -= 1
                elif ntrue[oc] == 1:
                    for jj in range(k):
                        l = cl[oc, jj]; w = abs(l)
                        if (a[w] == 1) == (l > 0):
                            crit[oc] = w; br[w] += 1; break
        if nu < best:
            best = nu
    return a, best

def load(path):
    cls = []; n = 0
    for line in open(path):
        if line.startswith(('p', 'c')) or not line.strip():
            if line.startswith('p'): n = int(line.split()[2])
            continue
        c = [int(x) for x in line.split() if x != '0']
        if c: cls.append(c)
    return cls, n

def worker(args):
    path, cb, seed, stop = args
    cls, n = load(path)
    k = max(len(c) for c in cls)
    cl = np.array([c + [c[0]] * (k - len(c)) for c in cls], dtype=np.int32)
    lists = [[] for _ in range(n + 1)]
    for i, c in enumerate(cls):
        for v in set(abs(x) for x in c): lists[v].append(i)
    occ_ptr = np.zeros(n + 2, np.int64)
    for v in range(n + 1): occ_ptr[v + 1] = occ_ptr[v] + len(lists[v])
    occ = np.array([i for L in lists for i in L], dtype=np.int32)
    rnd = 0
    while not stop.is_set():
        a, best = run(cl, k, n, occ_ptr, occ, cb, seed + rnd * 7919, 60_000_000); rnd += 1
        print(f"  core cb={cb:.2f} best={best}", flush=True)
        if best == 0:
            bits = ''.join('1' if a[v] else '0' for v in range(1, n + 1))
            if all(any((bits[abs(l)-1] == '1') == (l > 0) for l in c) for c in cls):
                return bits
    return None

if __name__ == '__main__':
    import gzip
    path = sys.argv[1]; tlimit = float(sys.argv[2]); base = int(sys.argv[3])
    if path.endswith('.gz'):
        open('/tmp/p.cnf','wb').write(gzip.open(path).read()); path='/tmp/p.cnf'
    ncores = mp.cpu_count()
    rng = random.Random(base)
    cbs = [rng.uniform(4.0, 5.6) for i in range(ncores)]
    print(f"{ncores} cores cbs={cbs}", flush=True)
    t0 = time.time()
    with mp.Manager() as mg:
        stop = mg.Event()
        pool = mp.Pool(ncores)
        it = pool.imap_unordered(worker, [(path, cb, rng.randint(1, 10**9), stop) for cb in cbs])
        while time.time() - t0 < tlimit:
            try:
                res = it.next(timeout=max(1, tlimit - (time.time() - t0)))
            except mp.TimeoutError:
                break
            except StopIteration:
                break
            if res:
                open('answer.txt', 'w').write(res); print("SOLVED", time.time() - t0, flush=True); break
        stop.set(); pool.terminate()
