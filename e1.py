# Load the data and libraries
import pandas as pd
import numpy as np
from scipy import stats
import pytest


ADULT_DF = pd.read_csv("adult_with_pii.csv")
adult_small = ADULT_DF.loc[:99, ["Education", "Marital Status", "Target"]].copy()
print(adult_small)
#EX1
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

#EX2
def _generalize_education_marital(df):
    """Apply the categorical generalization rules from Task 2 to a DataFrame."""
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
    married_status_values = {
        "Married-civ-spouse",
        "Married-spouse-absent",
        "Married-AF-spouse",
    }

    out = df.copy()
    if "Education" in out.columns:
        out["Education"] = np.where(
            out["Education"].isin(low_education_levels), "< HS", ">= HS"
        )
    if "Marital Status" in out.columns:
        out["Marital Status"] = np.where(
            out["Marital Status"].isin(married_status_values),
            "Married",
            "Not Married",
        )
    return out


def _suppression_mask(k, qis, df):
    """Return a boolean mask selecting rows in classes of size at least ``k``."""
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if df is None:
        raise ValueError("df must not be None.")
    if df.empty:
        return pd.Series(False, index=df.index)
    if len(qis) == 0:
        if len(df) >= k:
            return pd.Series(True, index=df.index)
        return pd.Series(False, index=df.index)

    missing_cols = [col for col in qis if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {missing_cols}")

    group_sizes = df.groupby(qis, dropna=False).size()
    size_series = (
        df[qis]
        .merge(group_sizes.rename("size").reset_index(), on=qis, how="left")["size"]
    )
    return size_series >= k


def suppress_rows(k, qis, df):
    """Return a copy of ``df`` with small equivalence classes removed."""
    mask = _suppression_mask(k, qis, df)
    return df[mask].copy()


def suppress_count(k, qis, df):
    """Return the number of rows that must be suppressed for k-anonymity."""
    mask = _suppression_mask(k, qis, df)
    return int((~mask).sum())


def generalize_categorical():
    """Generalize ``adult_small`` and suppress rows to achieve k=2 anonymity."""
    df = _generalize_education_marital(adult_small.copy())
    return suppress_rows(2, ["Education", "Marital Status"], df)



def test_generalize_categorical_produces_2_anonymous_dataset():
    gen_df = generalize_categorical()
    qis = ["Education", "Marital Status"]
    # After generalization and suppression, the resulting dataset must be 2-anonymous
    assert is_k_anonymous(2, qis, gen_df)


def test_is_k_anonymous_edge_cases_and_validation():
    df = pd.DataFrame({"A": [1, 1], "B": [2, 2]})

    with pytest.raises(ValueError):
        is_k_anonymous(0, ["A"], df)
    with pytest.raises(ValueError):
        is_k_anonymous(2, ["A"], None)
    with pytest.raises(KeyError):
        is_k_anonymous(1, ["missing"], df)

    # No QIs means one global class containing all rows.
    assert is_k_anonymous(2, [], df)
    assert not is_k_anonymous(3, [], df)
    assert not is_k_anonymous(1, ["A"], pd.DataFrame(columns=["A", "B"]))


#EX3
def generalize_numeric(zip, n):
    """Generalize a numeric value by replacing the last ``n`` digits with zeros."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer.")

    try:
        value = int(zip)
    except (TypeError, ValueError):
        raise ValueError("zip must be an integer or integer-like value.")

    if n == 0:
        return value

    factor = 10 ** n
    return (value // factor) * factor

def test_generalize_numeric_examples():
    assert generalize_numeric(47401, 0) == 47401
    assert generalize_numeric(47401, 2) == 47400
    assert generalize_numeric(47401, 4) == 40000


def test_generalize_numeric_validation_and_string_input():
    assert generalize_numeric("47401", 2) == 47400
    with pytest.raises(ValueError):
        generalize_numeric(47401, -1)
    with pytest.raises(ValueError):
        generalize_numeric("zip", 1)

#EX4
def make_adult_k_anonymous(k, zip_digits, age_digits):
    """Generalize full adult dataset (Zip, Age) and suppress for k-anonymity on
    QIs: Zip, Sex, Age.

    Returns a tuple: (k_anonymous_dataframe, suppressed_row_count).
    """
    if not isinstance(zip_digits, int) or zip_digits < 0:
        raise ValueError("zip_digits must be a non-negative integer.")
    if not isinstance(age_digits, int) or age_digits < 0:
        raise ValueError("age_digits must be a non-negative integer.")

    df = ADULT_DF.copy()
    df["Zip"] = df["Zip"].apply(lambda z: generalize_numeric(z, zip_digits))
    df["Age"] = df["Age"].apply(lambda a: generalize_numeric(a, age_digits))

    qis = ["Zip", "Sex", "Age"]
    suppressed = suppress_count(k, qis, df)
    df_k_anon = suppress_rows(k, qis, df)
    return df_k_anon, suppressed


def test_make_adult_k_anonymous_matches_suppress_count():
    # Same parameters should produce consistent suppression count and k-anonymous output.
    k = 3
    out_df, suppressed = make_adult_k_anonymous(k, zip_digits=2, age_digits=1)
    assert suppressed == (len(ADULT_DF) - len(out_df))
    assert is_k_anonymous(k, ["Zip", "Sex", "Age"], out_df)


def test_suppression_helpers_edge_cases_and_consistency():
    df = pd.DataFrame(
        {
            "QI1": ["a", "a", "b"],
            "QI2": ["x", "x", "y"],
            "S": [1, 2, 3],
        }
    )

    mask = _suppression_mask(2, ["QI1", "QI2"], df)
    assert mask.tolist() == [True, True, False]
    assert suppress_count(2, ["QI1", "QI2"], df) == 1
    kept = suppress_rows(2, ["QI1", "QI2"], df)
    assert len(kept) == 2

    # No QIs: keep all rows if len(df) >= k, otherwise suppress all.
    assert _suppression_mask(3, [], df).all()
    assert not _suppression_mask(4, [], df).any()
    assert _suppression_mask(1, ["QI1"], pd.DataFrame(columns=["QI1"])).empty

    with pytest.raises(ValueError):
        _suppression_mask(0, ["QI1"], df)
    with pytest.raises(ValueError):
        _suppression_mask(2, ["QI1"], None)
    with pytest.raises(KeyError):
        _suppression_mask(2, ["missing"], df)


def test_make_adult_k_anonymous_parameter_validation():
    with pytest.raises(ValueError):
        make_adult_k_anonymous(3, zip_digits=-1, age_digits=1)
    with pytest.raises(ValueError):
        make_adult_k_anonymous(3, zip_digits=1, age_digits=-1)



#EX5
def is_l_diverse(l, qis, sens_col, df, type="probabilistic"):
    """Check whether ``df`` is l-diverse for probabilistic or entropy variants."""
    if not isinstance(l, int) or l <= 0:
        raise ValueError("l must be a positive integer.")
    if df is None:
        raise ValueError("df must not be None.")
    if sens_col not in df.columns:
        raise KeyError(f"Sensitive column '{sens_col}' not found in DataFrame.")
    if type not in ("probabilistic", "entropy"):
        raise ValueError("type must be 'probabilistic' or 'entropy'.")
    if df.empty:
        return True

    if len(qis) == 0:
        groups = [df]
    else:
        missing_cols = [col for col in qis if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Columns not found in DataFrame: {missing_cols}")
        groups = [g for _, g in df.groupby(qis, dropna=False)]

    for g in groups:
        probs = g[sens_col].value_counts(dropna=False, normalize=True)
        if type == "probabilistic":
            if probs.max() > 1.0 / l:
                return False
        else:
            h = stats.entropy(probs.to_numpy())  # natural log base
            if h < np.log(l):
                return False
    return True


def test_is_l_diverse_probabilistic_and_entropy():
    # One QI group with sensitive distribution p(x)=0.5, p(y)=0.5
    df = pd.DataFrame(
        {
            "QI": ["a", "a", "a", "a"],
            "S": ["x", "x", "y", "y"],
        }
    )
    assert is_l_diverse(2, ["QI"], "S", df, type="probabilistic")
    assert not is_l_diverse(3, ["QI"], "S", df, type="probabilistic")
    assert is_l_diverse(2, ["QI"], "S", df, type="entropy")
    assert not is_l_diverse(3, ["QI"], "S", df, type="entropy")


def test_is_l_diverse_edge_cases_and_validation():
    df = pd.DataFrame({"QI": ["a", "a"], "S": ["x", "y"]})

    with pytest.raises(ValueError):
        is_l_diverse(0, ["QI"], "S", df)
    with pytest.raises(ValueError):
        is_l_diverse(2, ["QI"], "S", None)
    with pytest.raises(KeyError):
        is_l_diverse(2, ["QI"], "missing", df)
    with pytest.raises(ValueError):
        is_l_diverse(2, ["QI"], "S", df, type="other")
    with pytest.raises(KeyError):
        is_l_diverse(2, ["missing"], "S", df)

    # Empty dataframe is treated as l-diverse by definition in this implementation.
    empty_df = pd.DataFrame(columns=["QI", "S"])
    assert is_l_diverse(2, ["QI"], "S", empty_df, type="probabilistic")
    assert is_l_diverse(2, ["QI"], "S", empty_df, type="entropy")

    # With no QIs, diversity is checked on the whole table as one group.
    assert is_l_diverse(2, [], "S", df, type="probabilistic")
    assert is_l_diverse(2, [], "S", df, type="entropy")


#EX7
def max_l(qis, sens_col, df, variant="probabilistic"):
    """Return the largest integer l for which ``df`` is l-diverse.

    Parameters
    ----------
    qis : list[str]
        Quasi-identifier column names.
    sens_col : str
        Sensitive column name.
    df : pandas.DataFrame
        Dataset to evaluate.
    variant : {"probabilistic", "entropy"}
        l-diversity variant to use.
    """
    if df is None:
        raise ValueError("df must not be None.")
    if sens_col not in df.columns:
        raise KeyError(f"Sensitive column '{sens_col}' not found in DataFrame.")
    if variant not in ("probabilistic", "entropy"):
        raise ValueError("variant must be 'probabilistic' or 'entropy'.")
    if df.empty:
        return 0

    if len(qis) == 0:
        groups = [df]
    else:
        missing_cols = [col for col in qis if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Columns not found in DataFrame: {missing_cols}")
        groups = [g for _, g in df.groupby(qis, dropna=False)]

    per_group_limits = []
    for g in groups:
        probs = g[sens_col].value_counts(dropna=False, normalize=True)

        if variant == "probabilistic":
            p_max = float(probs.max())
            l_group = int(np.floor(1.0 / p_max)) if p_max > 0 else 0
        else:
            h = float(stats.entropy(probs.to_numpy()))  # natural log base
            l_group = int(np.floor(np.exp(h)))

        per_group_limits.append(l_group)

    if not per_group_limits:
        return 0
    return max(0, min(per_group_limits))


def generalize_full_adult_categorical():
    """Apply Question 2 categorical generalization rules to the full adult dataset."""
    return _generalize_education_marital(ADULT_DF.copy())


def test_max_l_matches_is_l_diverse_for_simple_case():
    df = pd.DataFrame(
        {
            "QI": ["a", "a", "a", "a"],
            "S": ["x", "x", "y", "y"],
        }
    )
    assert max_l(["QI"], "S", df, variant="probabilistic") == 2
    assert max_l(["QI"], "S", df, variant="entropy") == 2
    assert is_l_diverse(
        max_l(["QI"], "S", df, variant="probabilistic"),
        ["QI"],
        "S",
        df,
        type="probabilistic",
    )
    assert is_l_diverse(
        max_l(["QI"], "S", df, variant="entropy"),
        ["QI"],
        "S",
        df,
        type="entropy",
    )


def test_generalize_full_adult_categorical_values():
    gen_df = generalize_full_adult_categorical()
    assert set(gen_df["Education"].dropna().unique()).issubset({"< HS", ">= HS"})
    assert set(gen_df["Marital Status"].dropna().unique()).issubset(
        {"Married", "Not Married"}
    )


def test_max_l_edge_cases_and_validation():
    df = pd.DataFrame({"QI": ["a", "a", "a"], "S": ["x", "x", "y"]})

    with pytest.raises(ValueError):
        max_l(["QI"], "S", None)
    with pytest.raises(KeyError):
        max_l(["QI"], "missing", df)
    with pytest.raises(ValueError):
        max_l(["QI"], "S", df, variant="other")
    with pytest.raises(KeyError):
        max_l(["missing"], "S", df)

    assert max_l(["QI"], "S", pd.DataFrame(columns=["QI", "S"])) == 0
    # No QIs: evaluate diversity limits on a single global group.
    assert max_l([], "S", df, variant="probabilistic") == 1
    assert max_l([], "S", df, variant="entropy") == 1


if __name__ == "__main__":
    # Simple demonstration for Exercise 1
    print("adult_small head:")
    print(adult_small.head())

    qis_example = ["Education", "Marital Status"]
    print("\nChecking k-anonymity for adult_small using quasi-identifiers:", qis_example)
    for k in range(1, 11):
        print(f"k = {k}: {is_k_anonymous(k, qis_example, adult_small)}")

