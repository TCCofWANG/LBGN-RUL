

import sys
import os
import traceback
import torch


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))


class FakeArgs:

    num_nodes       = 14
    input_length    = 50

    hidden_dim      = 32
    dropout         = 0.1

    lstm_hid        = 32
    lstm_n_layers   = 3
    lstm_bid        = True

    K               = 4
    LCE_dim         = 16
    n_bands         = 4
    hop             = 2
    z_dim           = 1

    ndc_hidden      = 16

    d_model         = 32
    nhead           = 8
    nlayers         = 2

    nig_coeff       = 0.3

    device          = 'cpu'


MODELS = [

    ("working_model_RUL",    "Models_RUL.working_model_RUL",     "working_model_RUL"),
    ("PDMN",                 "Models_RUL.PDMN",                  "PDMN"),
    ("LBGN_RUL","Models_RUL.LBGN_RUL", "LBGN_RUL"),
    ("LBGN_RUL_ablation","Models_RUL.LBGN_RUL_ablation", "LBGN_RUL_ablation"),
    ("LBGN_RUL_v2","Models_RUL.LBGN_RUL_v2", "LBGN_RUL_v2"),
    ("NDC_PDMN",             "Models_RUL.NDC_PDMN",              "NDC_PDMN"),
    ("DAGCN_RUL",            "Models_RUL.DAGCN_RUL",             "DAGCN_RUL"),
    ("EviAdaptRUL",          "Models_RUL.EviAdaptRUL",           "EviAdaptRUL"),
    ("DAST_RUL",             "Models_RUL.DAST_RUL",              "DAST_RUL"),
    ("CADA_RUL",             "Models_RUL.CADA_RUL",              "CADA_RUL"),
    ("TACDA_RUL",            "Models_RUL.TACDA_RUL",             "TACDA_RUL"),
    ("OCS_DANN",             "Models_RUL.OCS_DANN",              "OCS_DANN"),
    ("MDAN_RUL",             "Models_RUL.MDAN_RUL",              "MDAN_RUL"),
    ("CCDG_RUL",             "Models_RUL.CCDG_RUL",              "CCDG_RUL"),
]

passed = []
failed = []

x   = torch.randn(2, 50, 14)
occ = torch.zeros(2, 50, 1)

print("=" * 60)
print("  Import + forward-pass check")
print("=" * 60)

for name, module_path, class_name in MODELS:
    try:
        mod   = __import__(module_path, fromlist=[class_name])
        cls   = getattr(mod, class_name)
        model = cls(FakeArgs())
        model.eval()
        with torch.no_grad():
            _, out = model(x, occ)
        assert 'gamma' in out, f"output dict missing 'gamma' key: {list(out.keys())}"
        assert out['gamma'].shape == (2, 1), f"unexpected pred shape {out['gamma'].shape}"
        print(f"  [OK]  {name}")
        passed.append(name)
    except Exception:
        err = traceback.format_exc().strip().splitlines()[-1]
        print(f"  [!!]  {name}")
        print(f"        {err}")
        failed.append((name, err))

print("=" * 60)
print(f"  {len(passed)} passed  /  {len(failed)} failed")
if failed:
    print("\n  Failed models:")
    for n, e in failed:
        print(f"    {n}: {e}")
    sys.exit(1)
else:
    print("\n  All imports OK.")
    sys.exit(0)
