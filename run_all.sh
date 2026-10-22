#!/usr/bin/env bash
# Full end-to-end reproduction of the ml_vs_fep pipeline.
# Runs every stage in dependency order and logs each step.

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"


ENV_NAME="${ML_V_FEP_ENV:-ML_v_FEP}"
ENVDIR="$(conda env list | awk -v n="$ENV_NAME" '$1==n{print $NF}')"
if [ -z "$ENVDIR" ] || [ ! -x "$ENVDIR/bin/python" ]; then
  echo "conda environment '$ENV_NAME' not found - run: conda env create -f ML_v_FEP.yaml" >&2
  exit 1
fi

export JUPYTER_PATH="$ENVDIR/share/jupyter"
JUPYTER_BIN="$ENVDIR/bin/jupyter"
PYTHON_BIN="$ENVDIR/bin/python"

stamp() { date '+%H:%M:%S'; }
note()  { echo "[$(stamp)] $*" | tee -a "$LOGDIR/00_SUMMARY.log"; }

run_py() {   # run_py <dir> <script> [args...]
  local d="$1" f="$2" tag; shift 2; tag=$(echo "${d}_${f}${*:+_$*}" | tr '/ ' '__')
  note "START py  $d/$f $*"
  ( cd "$ROOT/$d" && "$PYTHON_BIN" "$f" "$@" ) >"$LOGDIR/$tag.log" 2>&1 \
    && note "  OK   py  $d/$f $*" || note "  FAIL py  $d/$f $*  (see $tag.log)"
}

run_nb() {   
  local d="$1" f="$2" tag; tag=$(echo "${d}_${f}" | tr '/ ' '__')
  note "START nb  $d/$f"
  ( cd "$ROOT/$d" && "$JUPYTER_BIN" nbconvert --to notebook \
      --execute --inplace "$f" ) >"$LOGDIR/$tag.log" 2>&1 \
    && note "  OK   nb  $d/$f" || note "  FAIL nb  $d/$f  (see $tag.log)"
}

note "=============== RUN START ==============="

# --- clear resume caches so MIF/PYC and 3D really recompute -----------------
note "clearing resume caches"
rm -f "$ROOT"/A_MODELS/{MIF,PYC}/{RANDOM,STRAT}/comsia_log.json
rm -f "$ROOT"/A_MODELS/3D/{RANDOM,STRAT}/precomputed_descriptors.npy

# --- 1. base ML: 2D and 3D --------------------------------------------------
run_py A_MODELS/2D/RANDOM A_RF_MORGAN_COUNT.py
run_py A_MODELS/2D/STRAT  A_RF_MORGAN_COUNT.py
run_py A_MODELS/3D/RANDOM A_RF_3D_DESCRIPTORS.py
run_py A_MODELS/3D/STRAT  A_RF_3D_DESCRIPTORS_SIMPLE.py

# --- 2. 3D-QSAR (slow: PyComsia, 6 targets in parallel) ---------------------
run_nb A_MODELS/MIF/RANDOM A_QSAR_MIF.ipynb
run_nb A_MODELS/MIF/STRAT  A_QSAR_MIF.ipynb
run_nb A_MODELS/PYC/RANDOM A_QSAR_ORIGINAL_PYCOMSIA_FIELDS.ipynb
run_nb A_MODELS/PYC/STRAT  A_QSAR_ORIGINAL_PYCOMSIA_FIELDS.ipynb

# --- 3. reformat 3D-QSAR output into the shared JSON scaffold ---------------
for m in MIF PYC; do for s in RANDOM STRAT; do
  run_nb "A_MODELS/$m/$s" B_FORMAT_PREDICTIONS.ipynb
done; done

# --- 4. derive ddG from dG --------------------------------------------------
for m in 2D 3D; do for s in RANDOM STRAT; do
  run_nb "A_MODELS/$m/$s" B_DDG_FROM_DG.ipynb
done; done
for m in MIF PYC; do for s in RANDOM STRAT; do
  run_nb "A_MODELS/$m/$s" C_DDG_FROM_DG.ipynb
done; done

# --- 5. FEP reference (needs 2D's split scaffold) ---------------------------
run_nb A_MODELS/FEP/DG/OFF   OFF_RANDOM_GET_DG.ipynb
run_nb A_MODELS/FEP/DG/OFF   OFF_STRAT_GET_DG.ipynb
run_nb A_MODELS/FEP/DG/OPLS  OPLS_RANDOM_GET_DG.ipynb
run_nb A_MODELS/FEP/DG/OPLS  OPLS_STRAT_GET_DG.ipynb
run_nb A_MODELS/FEP/DDG/OFF  1_OFF_GRAPH.ipynb
run_nb A_MODELS/FEP/DDG/OPLS 1_OPLS_GRAPH.ipynb
run_nb A_MODELS/FEP/DDG/OFF  2_OFF_RANDOM_GET_DDG_FROM_GRAPH_FEP.ipynb
run_nb A_MODELS/FEP/DDG/OFF  "2_OFF_STRAT_GET_DDG_FROM_GRAPH_FEP_STRAT copy.ipynb"
run_nb A_MODELS/FEP/DDG/OPLS 2_OPLS_RANDOM_GET_DDG_FROM_GRAPH_FEP.ipynb
run_nb A_MODELS/FEP/DDG/OPLS 2_OPLS_STRAT_GET_DDG_FROM_GRAPH_FEP.ipynb
run_nb A_MODELS/FEP/DDG/Z_COMPARE_METHODS_OF_DDG_CALCULATION 3_COMPARE_DDG_METHODS.ipynb

# --- 6. meta-learner, then FUSION -------------------------------------------
for e in DG DDG; do for s in RANDOM STRAT; do
  run_nb "A_MODELS/Z_META/$e/$s" RIDGE.ipynb
done; done
run_nb A_MODELS/Z_META_FEP/DDG/RANDOM/OFF  RIDGE_WITH_FEP.ipynb
run_nb A_MODELS/Z_META_FEP/DDG/RANDOM/OPLS RIDGE_WITH_FEP.ipynb
run_nb A_MODELS/FUSION/DG  A_CONSENSUS_FEP.ipynb
run_nb A_MODELS/FUSION/DDG A_CONSENSUS_FEP.ipynb

# --- 7. analysis + figures (B_COMPARE_METHODS) ------------------------------
run_nb B_COMPARE_METHODS/DDG/EFFECT_SIZE_PLOTS COMPARE_STAT_RANDOM.ipynb
run_nb B_COMPARE_METHODS/DDG/EFFECT_SIZE_PLOTS COMPARE_STAT_RANDOMZ.ipynb
run_nb B_COMPARE_METHODS/DDG/EFFECT_SIZE_PLOTS COMPARE_STAT_STRAT.ipynb
run_nb B_COMPARE_METHODS/DG/EFFECT_SIZE_PLOTS  DG_COMPARE_STAT_RANDOM.ipynb
run_nb B_COMPARE_METHODS/DG/EFFECT_SIZE_PLOTS  DG_COMPARE_STAT_STRAT.ipynb
run_nb B_COMPARE_METHODS/DDG/CONSENSUS_FEP     OFF_CONSENSUS_BARPLOT.ipynb
run_nb B_COMPARE_METHODS/DDG/CONSENSUS_FEP     OPLS_CONSENSUS_BARPLOT.ipynb
run_nb B_COMPARE_METHODS/DDG/CONSENSUS_FEP     OFF_CONSENSUS_COMPARE.ipynb
run_nb B_COMPARE_METHODS/DDG/CONSENSUS_FEP     OPLS_CONSENSUS_COMPARE.ipynb
run_nb B_COMPARE_METHODS/DG/CONSENSUS_FEP      OFF_CONSENSUS_COMPARE.ipynb
run_nb B_COMPARE_METHODS/DG/CONSENSUS_FEP      OPLS_CONSENSUS_COMPARE.ipynb
run_nb B_COMPARE_METHODS/DDG/WINS              WINS_RANDOM_ML_VS_OFF.ipynb
run_nb B_COMPARE_METHODS/DDG/WINS              WINS_RANDOM_ML_VS_OPLS.ipynb
run_nb B_COMPARE_METHODS/DG/WINS               MET_OFF_DG_COMPARE_STAT_STRAT.ipynb
run_nb B_COMPARE_METHODS/DG/WINS               MET_OPLS_DG_COMPARE_STAT_STRAT.ipynb
run_nb B_COMPARE_METHODS/DDG/ABSOLUTE_SLOPE_ANALYSIS SLOPES_DDG_RAND.ipynb
run_nb B_COMPARE_METHODS/DDG/ABSOLUTE_SLOPE_ANALYSIS SLOPES_P_VS_TRUE_HEXBIN.ipynb
run_py B_COMPARE_METHODS/DDG/META_WITH_FEP COMPARE_META_WITH_FEP.py OFF
run_py B_COMPARE_METHODS/DDG/META_WITH_FEP COMPARE_META_WITH_FEP.py OPLS

note "=============== RUN COMPLETE ==============="
note "failures: $(grep -c 'FAIL' "$LOGDIR/00_SUMMARY.log")"
