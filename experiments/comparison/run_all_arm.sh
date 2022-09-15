while IFS= read -r MODEL_NAME; do
    echo "$MODEL_NAME"
    python fit_pymc_sadvi.py "$MODEL_NAME" advi
    python fit_pymc_sadvi.py "$MODEL_NAME" fullrank_advi
    python fit_raabbvi.py "$MODEL_NAME"
    python fit_dadvi.py "$MODEL_NAME"
    python fit_mcmc.py "$MODEL_NAME"
    python fit_dadvi_lrvb.py "$MODEL_NAME"
done < all_arm_names.txt


