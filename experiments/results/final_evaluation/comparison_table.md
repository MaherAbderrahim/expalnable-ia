| model | config | dataset | threshold | auc_roc | pr_auc | accuracy_at_threshold | sensitivity | specificity | precision | f1 | balanced_accuracy | ece_10bins | n_cases | n_positive | evaluated_on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PI-CAI official 5-fold UNet (pretrained) | official_default | val | 0.5000 | 0.8379 | 0.7852 | 0.7667 | 0.1765 | 1.0000 | 1.0000 | 0.3000 | 0.5882 | 0.1577 | 120 | 34 | validation |
| PI-CAI official 5-fold UNet (pretrained) | no_extract | val | 0.1572 | 0.8451 | 0.7893 | 0.8500 | 0.6176 | 0.9419 | 0.8077 | 0.7000 | 0.7798 | 0.1731 | 120 | 34 | validation |
| PI-CAI official 5-fold UNet (pretrained) | official_default | test | 0.5000 | 0.8042 | 0.7064 | 0.7833 | 0.2353 | 1.0000 | 1.0000 | 0.3810 | 0.6176 | 0.1445 | 120 | 34 | test(frozen) |
| PI-CAI official 5-fold UNet (pretrained) | no_extract | test | 0.1572 | 0.8081 | 0.7107 | 0.8083 | 0.5588 | 0.9070 | 0.7037 | 0.6230 | 0.7329 | 0.1680 | 120 | 34 | test(frozen) |