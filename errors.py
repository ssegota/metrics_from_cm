from sklearn.metrics import confusion_matrix


y_true = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
y_pred = [0, 1, 0, 0, 0, 1, 0, 1, 1, 1]
def_get_vals_from_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    return TN, FP, FN, TP

def get_accuracy(y_true, y_pred):
    TN, FP, FN, TP = def_get_vals_from_cm(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def get_precision(y_true, y_pred):
    TN, FP, FN, TP = def_get_vals_from_cm(y_true, y_pred)
    return TP / (TP + FP)

def get_recall(y_true, y_pred):
    TN, FP, FN, TP = def_get_vals_from_cm(y_true, y_pred)
    return TP / (TP + FN)

def get_specificity(y_true, y_pred):
    TN, FP, FN, TP = def_get_vals_from_cm(y_true, y_pred)
    return TN / (TN + FP)

def get_sensitive(y_true, y_pred):
    return get_recall(y_true, y_pred)

def get_f1_score(y_true, y_pred):
    precision = get_precision(y_true, y_pred)
    recall = get_recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)

def get_tpr(y_true, y_pred):
    return get_recall(y_true, y_pred)

def get_fpr(y_true, y_pred):
    TN, FP, FN, TP = def_get_vals_from_cm(y_true, y_pred)
    return FP / (TN + FP)

def get_tnr(y_true, y_pred):
    return get_specificity(y_true, y_pred)

def get_fnr(y_true, y_pred):
    return 1 - get_recall(y_true, y_pred)

def get_informedness(y_true, y_pred):
    return get_tpr(y_true, y_pred) + get_tnr(y_true, y_pred) - 1

def get_prevalence_threshold(y_true, y_pred):
    TN, FP, FN, TP = def_get_vals_from_cm(y_true, y_pred)
    return (TP + FN) / (TP + TN + FP + FN)

def get_for(y_true, y_pred):
    TN, FP, FN, TP = def_get_vals_from_cm(y_true, y_pred)
    return FN / (FN + TN)

def get_positive_likelihood_ratio(y_true, y_pred):
    return get_tpr(y_true, y_pred) / get_fpr(y_true, y_pred)

def get_negative_likelihood_ratio(y_true, y_pred):
    return get_fnr(y_true, y_pred) / get_tnr(y_true, y_pred)

def get_diagnostic_odds_ratio(y_true, y_pred):
    return get_positive_likelihood_ratio(y_true, y_pred) / get_negative_likelihood_ratio(y_true, y_pred)

def get_markedness(y_true, y_pred):
    return get_precision(y_true, y_pred) + get_for(y_true, y_pred) - 1

def get_ppv(y_true, y_pred):
    return get_precision(y_true, y_pred)

def get_prevalence(y_true, y_pred):
    TN, FP, FN, TP = def_get_vals_from_cm(y_true, y_pred)
    return (TP + FN) / (TP + TN + FP + FN)

def get_fdr(y_true, y_pred):
    TN, FP, FN, TP = def_get_vals_from_cm(y_true, y_pred)
    return FP / (FP + TP)

def get_npv(y_true, y_pred):
    TN, FP, FN, TP = def_get_vals_from_cm(y_true, y_pred)
    return TN / (TN + FN)

def get_balanced_accuracy(y_true, y_pred):
    return (get_tpr(y_true, y_pred) + get_tnr(y_true, y_pred)) / 2

def get_mcc(y_true, y_pred):
    TN, FP, FN, TP = def_get_vals_from_cm(y_true, y_pred)
    return (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5

def get_fowlkes_mallows_index(y_true, y_pred):
    TN, FP, FN, TP = def_get_vals_from_cm(y_true, y_pred)
    return TP / ((TP + FP) * (TP + FN)) ** 0.5

def get_jaccard_index(y_true, y_pred):
    TN, FP, FN, TP = def_get_vals_from_cm(y_true, y_pred)
    return TP / (TP + FP + FN)
