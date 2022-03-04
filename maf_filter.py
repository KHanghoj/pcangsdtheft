import sys
import gzip
import numpy as np
from  tools import shared_cy
from tools import reader_cy

def emMAF(L, iter=200, tole=1e-5, t=88):
    m = L.shape[0] # Number of sites
    f = np.empty(m, dtype=np.float32)
    f.fill(0.25) # Uniform initialization
    f_prev = np.copy(f)
    for i in range(iter):
        shared_cy.emMAF_update(L, f, t)
        diff = shared_cy.rmse1d(f, f_prev)
        if diff < tole:
            print("EM (MAF) converged at iteration: " + str(i+1))
            break
        f_prev = np.copy(f)
    return f


threads, maf_filter,fin,fout = sys.argv[1:]
maf_filter = float(maf_filter)
threads = int(threads)
data = reader_cy.readBeagle(fin)
mafs = emMAF(data, t=threads)
np.savetxt(fout.replace(".gz", ".mafs.gz"), mafs, fmt="%6f")
maf_keep = np.logical_and(mafs>=maf_filter, mafs<=(1-maf_filter))

with gzip.open(fout, 'wb') as fhout, gzip.open(fin, 'rb') as fhin, gzip.open(fout.replace(".gz", "")+".trashed.gz", 'wb') as fhtrash:
    fhout.write(next(fhin))
    for idx, line in enumerate(fhin):
        if maf_keep[idx]:
            fhout.write(line)
        else:
            fhtrash.write(line)
