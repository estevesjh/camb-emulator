# CAMB + CosmoSIS

Generate LHS cosmologies, run CAMB via CosmoSIS, and plot power spectra.

1. **Generate LHS parameters**

```bash
python make_test_lhs_params.py
```

Creates `test_LHS_CAMB.npz`  
Example `omega_m`:

```
[0.5236, 0.3060, 0.3778, 0.7519, ...]
```

2. **Convert to CosmoSIS list**

```bash
python make_test_lhs_params_list.py
```

Creates `test_LHS_CAMB.list` ready for CosmoSIS.

3. **Run CAMB via CosmoSIS**

```bash
cosmosis camb_pipeline.ini
```

Outputs:

```
./output/
linear.dat   boost.dat   k_modes.txt
```

- Rows in `linear.dat`: `N_rows = N_cosmologies × N_z` → 10 × 50 = 500  
- Columns: `N_columns = N_cosmo_params + 1 + N_k` → 1 + 1 + 150 = 152

4. **Plot the spectra**

```bash
python check_camb_spectra_10.py
```

Generates `test_camb_spectra_10.pdf`  

Prints:

```
k-modes #: 150
Pk len: 150
```

**Files:**

- `make_test_lhs_params.py` → generate `.npz` LHS parameters  
- `make_test_lhs_params_list.py` → convert `.npz` to `.list`  
- `camb_pipeline.ini` → CosmoSIS CAMB config  
- `save_pk.py` → post-processing to saves CAMB power spectra  
- `check_camb_spectra_10.py` → plots spectra  
- `linear.dat`, `boost.dat`, `k_modes.txt` → CAMB output
