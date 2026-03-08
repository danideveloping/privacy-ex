# Load the data and libraries
import pandas as pd
import numpy as np
import pytest


# Load the full Adult dataset with PII (assumed to be in the same directory)
ADULT_DF = pd.read_csv("adult_with_pii.csv")

# Subset used in Exercise 1: first 100 rows and selected columns
adult_small = ADULT_DF.loc[:99, ["Education", "Marital Status", "Target"]].copy()


def is_k_anonymous(k, qis, df):
    """Return True iff ``df`` satisfies k-anonymity for the given quasi-identifiers.

    Parameters
    ----------
    k : int
        The k parameter in k-anonymity. All equivalence classes induced by ``qis``
        must have size at least ``k``.
    qis : list of str
        Column names in ``df`` that act as quasi-identifiers.
    df : pandas.DataFrame
        Dataset to be checked.
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    if df is None:
        raise ValueError("df must not be None.")

    if len(qis) == 0:
        # With no quasi-identifiers, all rows are in a single equivalence class
        return len(df) >= k

    # Ensure that all quasi-identifier columns exist in the dataframe
    missing_cols = [col for col in qis if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {missing_cols}")

    if df.empty:
        # Empty datasets trivially cannot satisfy k-anonymity for k >= 1
        return False

    # Group by the quasi-identifiers and compute the size of each equivalence class
    group_sizes = df.groupby(qis, dropna=False).size()

    # k-anonymity requires that every equivalence class has size at least k
    return (group_sizes >= k).all()


def generalize_categorical():
    """Generalize categorical attributes in ``adult_small`` and suppress rows to
    achieve 2-anonymity, assuming ``Target`` is NOT a quasi-identifier.

    Generalization rules:
    - ``Education``: levels below ``HS-grad`` become ``< HS``, all others ``>= HS``.
    - ``Marital Status``: values in a married state become ``Married``, all others
      become ``Not Married``.

    Suppression:
    - After generalization, delete rows that are in equivalence classes (by the
      generalized quasi-identifiers) of size < 2.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with generalized quasi-identifiers and suppressed rows.
    """
    df = adult_small.copy()

    df = _generalize_education_marital(df)

    # Suppress rows to achieve k=2 anonymity using generalized QIs
    qis = ["Education", "Marital Status"]
    df_suppressed = suppress_rows(2, qis, df)

    return df_suppressed


def _generalize_education_marital(df):
    """Apply the categorical generalization rules from Task 2 to a DataFrame."""
    # Generalize Education: below HS-grad vs others
    low_education_levels = {
        "Preschool",
        "1st-4th",
        "5th-6th",
        "7th-8th",
        "9th",
        "10th",
        "11th",
        "12th",
    }
    df = df.copy()
    if "Education" in df.columns:
        df["Education"] = np.where(
            df["Education"].isin(low_education_levels), "< HS", ">= HS"
        )

    # Generalize Marital Status: Married vs Not Married
    married_status_values = {
        "Married-civ-spouse",
        "Married-spouse-absent",
        "Married-AF-spouse",
    }
    if "Marital Status" in df.columns:
        df["Marital Status"] = np.where(
            df["Marital Status"].isin(married_status_values),
            "Married",
            "Not Married",
        )

    return df


def generalize_numeric(zip, n):
    """Generalize a numeric value by replacing the last ``n`` digits with zeros.

    Examples (required by the exercise):
    - generalize_numeric(47401, 0) == 47401
    - generalize_numeric(47401, 2) == 47400
    - generalize_numeric(47401, 4) == 40000
    """
    if n < 0:
        raise ValueError("n must be non-negative.")

    try:
        value = int(zip)
    except (TypeError, ValueError):
        raise ValueError("zip must be an integer or integer-like value.")

    if n == 0:
        return value

    factor = 10 ** n
    return (value // factor) * factor


def _suppression_mask(k, qis, df):
    """Return a boolean mask selecting rows that belong to equivalence classes
    of size at least ``k`` for the given quasi-identifiers."""
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    if df is None:
        raise ValueError("df must not be None.")

    if df.empty:
        return pd.Series(False, index=df.index)

    if len(qis) == 0:
        # Either keep all rows (if dataset big enough) or suppress all
        if len(df) >= k:
            return pd.Series(True, index=df.index)
        return pd.Series(False, index=df.index)

    missing_cols = [col for col in qis if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {missing_cols}")

    # Compute group sizes for each equivalence class
    group_sizes = df.groupby(qis, dropna=False).size()

    # Build a mask indicating rows whose (qis) tuple is in one of the valid keys
    size_series = (
        df[qis]
        .merge(
            group_sizes.rename("size").reset_index(),
            on=qis,
            how="left",
        )["size"]
    )
    mask = size_series >= k
    return mask


def suppress_rows(k, qis, df):
    """Return a copy of ``df`` with rows in small equivalence classes removed."""
    mask = _suppression_mask(k, qis, df)
    return df[mask].copy()


def suppress_count(k, qis, df):
    """Return the number of rows that must be suppressed to achieve k-anonymity."""
    mask = _suppression_mask(k, qis, df)
    # mask == True → kept rows, False → suppressed rows
    return int((~mask).sum())


def is_l_diverse(l, qis, sens_col, df, type='probabilistic'):
    """Check whether ``df`` is l-diverse for the given quasi-identifiers and
    sensitive column.

    Parameters
    ----------
    l : int
        Diversity parameter.
    qis : list of str
        Quasi-identifier columns.
    sens_col : str
        Sensitive attribute column name.
    df : pandas.DataFrame
        Dataset to be checked.
    type : {'probabilistic', 'entropy'}
        Variant of l-diversity.
    """
    if not isinstance(l, int) or l <= 0:
        raise ValueError("l must be a positive integer.")

    if df is None:
        raise ValueError("df must not be None.")

    if sens_col not in df.columns:
        raise KeyError(f"Sensitive column '{sens_col}' not found in DataFrame.")

    if type not in ("probabilistic", "entropy"):
        raise ValueError("type must be 'probabilistic' or 'entropy'.")

    if df.empty:
        # Vacuously l-diverse: there are no groups that violate the definition
        return True

    # Group by QIs; if qis is empty, treat the whole df as one group
    if len(qis) == 0:
        groups = [df]
    else:
        missing_cols = [col for col in qis if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Columns not found in DataFrame: {missing_cols}")
        groups = [g for _, g in df.groupby(qis, dropna=False)]

    for g in groups:
        counts = g[sens_col].value_counts(dropna=False)
        total = float(len(g))
        if total == 0:
            continue
        probs = counts / total

        if type == "probabilistic":
            # Probabilistic l-diversity: max probability ≤ 1/l
            if probs.max() > 1.0 / l:
                return False
        else:
            # Entropy l-diversity: H(G) ≥ ln(l)
            # Use natural log for entropy
            p = probs.to_numpy()
            entropy = -np.sum(p * np.log(p))
            if entropy < np.log(l):
                return False

    return True


def max_l(qis, sens_col, df, type='probabilistic'):
    """Return the largest integer l for which ``df`` is l-diverse.

    Parameters
    ----------
    qis : list of str
        Quasi-identifier columns.
    sens_col : str
        Sensitive attribute column name.
    df : pandas.DataFrame
        Dataset to be checked.
    type : {'probabilistic', 'entropy'}
        Variant of l-diversity.
    """
    if df is None:
        raise ValueError("df must not be None.")

    if sens_col not in df.columns:
        raise KeyError(f"Sensitive column '{sens_col}' not found in DataFrame.")

    if type not in ("probabilistic", "entropy"):
        raise ValueError("type must be 'probabilistic' or 'entropy'.")

    if df.empty:
        # No constraint from any group; define max l as 0 (no meaningful diversity)
        return 0

    # Group by QIs; if qis is empty, treat the whole df as one group
    if len(qis) == 0:
        grouped = [(None, df)]
    else:
        missing_cols = [col for col in qis if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Columns not found in DataFrame: {missing_cols}")
        grouped = list(df.groupby(qis, dropna=False))

    l_limits = []

    for _, g in grouped:
        counts = g[sens_col].value_counts(dropna=False)
        total = float(len(g))
        if total == 0:
            continue
        probs = counts / total

        if type == "probabilistic":
            p_max = probs.max()
            # We need max_i p_i ≤ 1/l ⇒ l ≤ 1/p_max
            l_group = int(np.floor(1.0 / p_max)) if p_max > 0 else 0
        else:
            # Entropy variant: H(G) ≥ ln(l) ⇒ l ≤ exp(H(G))
            p = probs.to_numpy()
            entropy = -np.sum(p * np.log(p))
            l_group = int(np.floor(np.exp(entropy)))

        l_limits.append(l_group)

    if not l_limits:
        return 0

    # The dataset is l-diverse only for l up to the minimum group limit
    return max(0, min(l_limits))


####################
# Pytest test cases
####################


def test_is_k_anonymous_simple_true():
    # Simple case where each combination appears at least twice
    data = {
        "A": ["x", "x", "y", "y"],
        "B": [1, 1, 2, 2],
        "S": [10, 20, 30, 40],  # sensitive column (ignored by the function)
    }
    df = pd.DataFrame(data)
    assert is_k_anonymous(2, ["A", "B"], df) is True


def test_is_k_anonymous_simple_false():
    # One equivalence class has size 1, so not 2-anonymous
    data = {
        "A": ["x", "x", "y"],
        "B": [1, 1, 2],
    }
    df = pd.DataFrame(data)
    assert is_k_anonymous(2, ["A", "B"], df) is False


def test_is_k_anonymous_empty_qis():
    # With empty qis, all rows form a single equivalence class
    df = pd.DataFrame({"A": [1, 2, 3]})
    assert is_k_anonymous(1, [], df) is True
    assert is_k_anonymous(3, [], df) is True
    assert is_k_anonymous(4, [], df) is False


def test_is_k_anonymous_uses_all_rows_with_nans():
    # NaN values should still contribute to an equivalence class
    df = pd.DataFrame(
        {
            "A": ["x", "x", np.nan, np.nan],
            "B": [1, 1, 2, 2],
        }
    )
    # Each (A,B) combination (including NaN) appears twice
    assert is_k_anonymous(2, ["A", "B"], df) is True


def test_is_k_anonymous_invalid_k_raises():
    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(ValueError):
        is_k_anonymous(0, ["A"], df)
    with pytest.raises(ValueError):
        is_k_anonymous(-1, ["A"], df)


def test_is_k_anonymous_missing_column_raises():
    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(KeyError):
        is_k_anonymous(1, ["A", "B"], df)


def test_is_k_anonymous_empty_df_is_never_k_anonymous():
    df = pd.DataFrame(columns=["A", "B"])
    assert is_k_anonymous(1, ["A"], df) is False


def test_is_k_anonymous_on_adult_small():
    # Using the subset specified in the exercise
    qis = ["Education", "Marital Status"]
    assert is_k_anonymous(1, qis, adult_small) is True
    assert is_k_anonymous(2, qis, adult_small) is False


def test_suppress_rows_removes_small_equivalence_classes():
    data = {
        "A": ["x", "x", "y", "z"],
        "B": [1, 1, 2, 3],
    }
    df = pd.DataFrame(data)
    # For k=2, only the (x,1) class (size 2) should remain
    result = suppress_rows(2, ["A", "B"], df)
    assert len(result) == 2
    assert (result["A"] == "x").all()
    assert (result["B"] == 1).all()


def test_suppress_count_empty_qis_behavior():
    df = pd.DataFrame({"A": [1, 2, 3]})
    # For k <= len(df), no rows need to be suppressed
    assert suppress_count(3, [], df) == 0
    # For k > len(df), all rows must be suppressed
    assert suppress_count(4, [], df) == 3


def test_generalize_categorical_produces_2_anonymous_dataset():
    gen_df = generalize_categorical()
    qis = ["Education", "Marital Status"]
    # After generalization and suppression, the resulting dataset must be 2-anonymous
    assert is_k_anonymous(2, qis, gen_df) is True


def test_generalize_numeric_examples():
    assert generalize_numeric(47401, 0) == 47401
    assert generalize_numeric(47401, 2) == 47400
    assert generalize_numeric(47401, 4) == 40000


def test_is_l_diverse_probabilistic_and_entropy():
    # Single group defined by empty QIs
    df = pd.DataFrame(
        {
            "QI": ["a", "a", "a", "a"],
            "S": ["x", "x", "y", "y"],  # p(x)=0.5, p(y)=0.5
        }
    )
    # Probabilistic: max p = 0.5, so l <= 2
    assert is_l_diverse(2, ["QI"], "S", df, type="probabilistic") is True
    assert is_l_diverse(3, ["QI"], "S", df, type="probabilistic") is False

    # Entropy: H = ln(2), so l <= 2
    assert is_l_diverse(2, ["QI"], "S", df, type="entropy") is True
    assert is_l_diverse(3, ["QI"], "S", df, type="entropy") is False


def test_max_l_matches_is_l_diverse_for_simple_case():
    df = pd.DataFrame(
        {
            "QI": ["a", "a", "a", "a"],
            "S": ["x", "x", "y", "y"],
        }
    )
    # For this dataset, the largest l for both variants should be 2
    assert max_l(["QI"], "S", df, type="probabilistic") == 2
    assert max_l(["QI"], "S", df, type="entropy") == 2


if __name__ == "__main__":
    # Simple demonstration for Exercise 1
    print("adult_small head:")
    print(adult_small.head())

    qis_example = ["Education", "Marital Status"]
    print("\nChecking k-anonymity for adult_small using quasi-identifiers:", qis_example)
    for k in range(1, 11):
        print(f"k = {k}: {is_k_anonymous(k, qis_example, adult_small)}")

