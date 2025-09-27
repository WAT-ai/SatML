import os, yaml
from copy import deepcopy
import argparse, pprint

def _set_by_dotted_key(d, dotted, value):
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def apply_overrides(cfg: dict, pairs: list[str]) -> dict:
    """pairs like ['train.epochs=10', 'train.optimizer.lr=1e-4']"""
    out = deepcopy(cfg)
    for p in pairs or []:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        # try to cast numbers/bools
        vv = v
        if v.lower() in ("true", "false"):
            vv = v.lower() == "true"
        else:
            try:
                vv = int(v)
            except ValueError:
                try:
                    vv = float(v)
                except ValueError:
                    pass
        _set_by_dotted_key(out, k, vv)
    return out

def load_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # simple ${var} interpolation for two common cases
    exp = cfg.get("experiment_name", "exp")
    def subst(s: str) -> str:
        return (s.replace("${experiment_name}", exp)
                 if isinstance(s, str) else s)
    def walk(x):
        if isinstance(x, dict):
            return {k: walk(subst(v)) for k,v in x.items()}
        if isinstance(x, list):
            return [walk(subst(v)) for v in x]
        return subst(x)
    cfg = walk(cfg)
    # MLflow URI: env wins
    ml_uri_env = os.getenv("MLFLOW_TRACKING_URI")
    if ml_uri_env:
        cfg.setdefault("mlflow", {})["tracking_uri"] = ml_uri_env
    # cli overrides
    return apply_overrides(cfg, overrides)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/defaults.yaml")
    ap.add_argument("--set", action="append", dest="sets",
                    help="override like key.sub=val", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.sets)
    pprint.pp(cfg)
    print("\nOK: loaded config and overrides.")
