#!/bin/bash
#SBATCH --job-name=merge_v2_slc
#SBATCH --qos=shared
#SBATCH -C cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --account=des
#SBATCH --output=logs/merge_v2_slc_%j.out
#SBATCH --error=logs/merge_v2_slc_%j.err

# Merge v2 array slices: slice{NNN}_linear_v2.dat -> linear_v2.dat
# and slice{NNN}_linear_nonu_v2.dat -> linear_nonu_v2.dat
# Also: pick slice000_k_modes_v2.txt as the canonical k_modes_v2.txt
# Finally: head/tail split 90/10 into _train.dat / _test.dat

cd "${SLURM_SUBMIT_DIR:-$PSCRATCH/camb-emulator/camb-for-cp}"

source /global/common/software/des/jesteves/cosmosis/setup-cosmosis-nersc \
    /global/common/software/des/common/Conda_Envs/y3cl_je

echo "=== v2 slice merge ==="
date

# k_modes: all slices produce the same grid, so just take slice 000
if [ -f slice000_k_modes_v2.txt ]; then
    cp -f slice000_k_modes_v2.txt k_modes_v2.txt
    echo "Copied slice000_k_modes_v2.txt -> k_modes_v2.txt"
fi

# Integrity check: linear and linear_nonu row counts must match per slice,
# otherwise train/test splits by position will desync the two files.
echo ""
echo "=== per-slice row count check ==="
python <<'PY'
import glob, sys
bad = []
for lf in sorted(glob.glob('slice*_linear_v2.dat')):
    nf = lf.replace('_linear_v2.dat', '_linear_nonu_v2.dat')
    try:
        with open(lf,'rb') as f: nl = f.read().count(b'\n')
        with open(nf,'rb') as f: nn = f.read().count(b'\n')
    except FileNotFoundError:
        bad.append((lf, 'nonu file missing'))
        continue
    if nl != nn:
        bad.append((lf, f'linear={nl} nonu={nn}'))
if bad:
    print(f'ERROR: {len(bad)} slices have mismatched linear/nonu row counts:')
    for f, err in bad[:10]:
        print(f'  {f}: {err}')
    sys.exit(1)
print('All slices have matching linear/nonu row counts.')
PY

# Merge both linear_v2 and linear_nonu_v2 via streaming cat
python scripts/merge_pk_outputs_parallel.py \
    --glob-template 'slice*_{name}.dat' \
    --output-template '{name}.dat' \
    --names linear_v2 linear_nonu_v2

# Sanity check on merged files
n_lin=$(wc -l < linear_v2.dat)
n_non=$(wc -l < linear_nonu_v2.dat)
if [ "$n_lin" != "$n_non" ]; then
    echo "ERROR: merged row counts differ: linear=$n_lin nonu=$n_non"
    exit 1
fi
echo "Merged counts OK: $n_lin rows each"

echo ""
echo "=== 90/10 train/test split ==="
for name in linear_v2 linear_nonu_v2; do
    total=$(wc -l < ${name}.dat)
    n_train=$(( total * 9 / 10 ))
    n_test=$(( total - n_train ))
    echo "${name}.dat: total=${total}, train=${n_train}, test=${n_test}"
    head -n ${n_train} ${name}.dat > ${name}_train.dat
    tail -n ${n_test}  ${name}.dat > ${name}_test.dat
done

echo ""
date
echo "Finished merge+split"
