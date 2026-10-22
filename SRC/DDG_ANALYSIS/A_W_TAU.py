""""
Copyright (C) 2025 ETH Zurich, Stefan J.P. Chrin, Gregory Landrum, Sereina Riniker
"""

import numpy as np

def get_tie_counts(sxps,tol):
    npts = len(sxps)
    tx = []
    currv = sxps[0][0]
    tcount = 0
    for i in range(1,npts):
        xi = sxps[i][0]
        if xi-currv < tol:
            tcount += 1
        else:
            currv = xi
            if tcount:
                tx.append(tcount)
    return tx
    
# uses the scipy definition (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html)
def window_tau(xps,yps,tol=0.0, xtol=None, ytol=None,
              silent=False):
    assert len(xps)==len(yps)
    if xtol is None:
        xtol = tol
    if ytol is None:
        ytol = tol
    sxps = sorted((x,i) for i,x in enumerate(xps))
    oyps = [yps[i] for x,i in sxps]
    nconcord = 0
    ndiscord = 0
    npts = len(sxps)
    ntx = 0
    for i,xi in enumerate(sxps):
        xi = xi[0]
        yi = oyps[i]
        for j in range(i+1,npts):
            xj = sxps[j][0]
            yj = oyps[j]
            dx = xj-xi
            dy = yj-yi
            # we know xj >= xi because of the sort
            if dx>xtol and abs(dy)>ytol:
                if dy>0:
                    nconcord += 1
                else:
                    ndiscord += 1
            elif dx<=tol and abs(dy)>tol:
                # tie in x, but not in y:
                ntx += 1
    nty = 0
    # find ties in y, but not in x:
    for i,yi in enumerate(yps):
        for j in range(i+1,npts):
            if abs(yps[j] - yi)<=ytol:
                if abs(xps[j] - xps[i]) > xtol:
                    nty += 1

    if not silent:
        print('\t',nconcord,ndiscord,ntx,nty)
    tau = (nconcord - ndiscord) / np.sqrt((nconcord + ndiscord + ntx)*(nconcord + ndiscord + nty))
    return tau