MODEL_NAME=microcredit

# TODO: Add SADVI, DADVI + LRVB
# python fit_raabbvi.py "$MODEL_NAME"
# python fit_dadvi.py "$MODEL_NAME"
# python fit_mcmc.py "$MODEL_NAME"
python fit_dadvi_lrvb.py "$MODEL_NAME"
