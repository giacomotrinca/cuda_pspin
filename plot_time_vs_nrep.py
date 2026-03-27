#!/usr/bin/env python3
"""Plot time per MC iteration vs nrep at fixed N, averaging over samples."""

import os
import re
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt

# Default size; override with command line argument
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24

# Collect data from data/N{N}_NR*_S*/time_per_iter.txt  and  data/N{N}_NR*/time_per_iter.txt
pattern = f"data/N{N}_NR*"
dirs = sorted(glob.glob(pattern))

if not dirs:
    print(f"No data directories found matching {pattern}")
    sys.exit(1)

# Parse nrep and sample from directory names
# Formats: N24_NR8_S0, N24_NR8 (legacy, no sample label)
data = {}  # nrep -> list of ms_per_iter values (one per sample)

for d in dirs:
    timefile = os.path.join(d, "time_per_iter.txt")
    if not os.path.isfile(timefile):
        continue
    basename = os.path.basename(d)
    m = re.match(rf"N{N}_NR(\d+)(?:_S(\d+))?$", basename)
    if not m:
        continue
    nrep = int(m.group(1))
    with open(timefile) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                ms = float(parts[4])
                data.setdefault(nrep, []).append(ms)

if not data:
    print("No time_per_iter.txt data found")
    sys.exit(1)

nreps = sorted(data.keys())
means = []
errs = []

print(f"N = {N}")
print(f"{'nrep':>6s}  {'samples':>7s}  {'<ms/iter>':>10s}  {'err':>10s}")
print("-" * 40)
for nr in nreps:
    vals = np.array(data[nr])
    m = vals.mean()
    e = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    means.append(m)
    errs.append(e)
    print(f"{nr:6d}  {len(vals):7d}  {m:10.4f}  {e:10.4f}")

nreps = np.array(nreps)
means = np.array(means)
errs = np.array(errs)

# Plot
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(nreps, means, yerr=errs, fmt="o-", color="C0",
            linewidth=1.5, markersize=6, capsize=3)
ax.set_xlabel("nrep", fontsize=13)
ax.set_ylabel("ms / iteration", fontsize=13)
ax.set_title(f"Time per MC iteration  (N = {N})", fontsize=14)
ax.set_xscale("log", base=2)
ax.set_xticks(nreps)
ax.set_xticklabels([str(n) for n in nreps])
ax.grid(True, alpha=0.3)
fig.tight_layout()

outfile = f"time_vs_nrep_N{N}.png"
fig.savefig(outfile, dpi=150)
print(f"\nPlot saved to {outfile}")
plt.show()
