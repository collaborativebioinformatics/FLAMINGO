# Running simmrd against these parameter files

simmrd is not vendored here. Clone it next to this repo and use its CLI:

```bash
git clone https://github.com/noahlorinczcomi/simmrd ../simmrd
cd ../simmrd/cli
pixi install && pixi run setup
for p in simmrd/params/*.yaml; do
  pixi run simulate --params "$p" --output "$(basename "${p%.yaml}").rds" \
    --iterations 500 --seed 42
done
```

Each YAML maps directly onto `simmrd::set_params()` arguments. Omitted keys
take the package defaults.

| File | Scenario |
|---|---|
| `params/valid.yaml` | No pleiotropy, no overlap. Sanity baseline. |
| `params/uhp.yaml` | 20% uncorrelated horizontal pleiotropy. |
| `params/chp.yaml` | 20% correlated horizontal pleiotropy (InSIDE violated). |
| `params/uhp_chp.yaml` | 10% UHP + 10% CHP, full sample overlap. |
| `params/weak.yaml` | Valid instruments, mean F fixed at 10. |
