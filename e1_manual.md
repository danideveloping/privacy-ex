## Exercise 1 – k-Anonymity and Variants (Manual & Conceptual Answers)

### 1. How to run the code

- **Run the main script** (demonstration for Exercise 1, k-anonymity on `adult_small`):

```bash
python e1.py
```

This will:
- Load `adult_with_pii.csv`.
- Construct `adult_small` as the first 100 rows with columns `Education`, `Marital Status`, `Target`.
- Print the head of `adult_small`.
- Check and print whether `adult_small` satisfies k-anonymity for `k = 1..10` using quasi-identifiers `["Education", "Marital Status"]`.

- **Run the tests** (requires `pytest` to be installed):

```bash
pytest e1.py
```

This executes all test cases defined at the bottom of `e1.py`.

### 2. How to interpret the outputs

- **From `python e1.py`**:
  - The printed table labeled `adult_small head` shows sample rows of the subset used in the exercise.
  - The subsequent lines `k = X: True/False` indicate whether `adult_small` is k-anonymous for that value of `k` with respect to the quasi-identifiers `Education` and `Marital Status`.
    - `True` means every equivalence class (group of records sharing the same values of the quasi-identifiers) has at least `k` records.
    - `False` means there exists at least one equivalence class of size `< k`.

- **From `pytest e1.py`**:
  - All tests **passing** means:
    - `is_k_anonymous` behaves correctly on normal cases and edge cases.
    - `suppress_count` deletes exactly the rows needed so that all remaining equivalence classes have size at least `k`.
    - `generalize_categorical` correctly generalizes and suppresses `adult_small` so that the resulting dataset is 2-anonymous.
  - A failing test will show which scenario (function and input shape) violates the expected behavior.

### 3. Test case rationale

The tests are designed to cover both **typical** and **edge** scenarios for each implemented function:

- **`is_k_anonymous` tests**:
  - `test_is_k_anonymous_simple_true`: simple dataset where each quasi-identifier combination appears at least twice. Verifies the positive case for `k = 2`.
  - `test_is_k_anonymous_simple_false`: includes a combination of quasi-identifiers that appears only once. Verifies the negative case where k-anonymity is violated.
  - `test_is_k_anonymous_empty_qis`: covers the special case where the list of quasi-identifiers is empty. Here, all rows form a single equivalence class; we check behavior for `k` equal to and greater than the dataset size.
  - `test_is_k_anonymous_uses_all_rows_with_nans`: ensures that rows containing `NaN` in quasi-identifiers are still counted toward equivalence classes.
  - `test_is_k_anonymous_invalid_k_raises`: checks input validation for `k <= 0`, which must raise a `ValueError`.
  - `test_is_k_anonymous_missing_column_raises`: verifies that missing quasi-identifier columns trigger a `KeyError`.
  - `test_is_k_anonymous_empty_df_is_never_k_anonymous`: tests the behavior on an empty DataFrame.
  - `test_is_k_anonymous_on_adult_small`: uses the actual `adult_small` subset from the exercise to confirm that it is 1-anonymous but not 2-anonymous for quasi-identifiers `Education` and `Marital Status`.

  These tests together cover:
  - Normal use (k-anonymous and not k-anonymous).
  - Edge conditions: empty QI list, empty dataset, NaNs.
  - Invalid inputs: wrong `k`, missing columns.
  - Realistic dataset: `adult_small`.

- **`suppress_count` tests**:
  - `test_suppress_count_removes_small_equivalence_classes`: checks that the function suppresses only those rows in small equivalence classes, leaving intact classes where `size >= k`.
  - `test_suppress_count_empty_qis_behavior`: exercises the special case where there are no quasi-identifiers. The entire dataset is one equivalence class, so:
    - For `k <= len(df)`, all rows must be kept.
    - For `k > len(df)`, all rows must be suppressed.

  Together these verify both normal grouping behavior and the empty-QI corner case.

- **`generalize_categorical` tests**:
  - `test_generalize_categorical_produces_2_anonymous_dataset`: runs the complete generalization + suppression pipeline on `adult_small` and then calls `is_k_anonymous(2, ["Education", "Marital Status"], gen_df)`. This checks the overall requirement for Task 2: the output must be 2-anonymous.

This combination of tests provides **high coverage**:
- Every implemented function is executed.
- Branches for valid/invalid arguments and corner cases are tested.
- The functions are also exercised on the actual exercise dataset, confirming realistic behavior beyond synthetic toy examples.

---

## Conceptual Answers

### 4. Identifiers, quasi-identifiers, and sensitive attributes (Task 1)

We work with:

- Full dataset: `adult_with_pii.csv`.
- Subset for Exercise 1:

```text
adult_small = first 100 rows of adult_with_pii with columns
              ["Education", "Marital Status", "Target"]
```

- **Identifiers in `adult_small`**:
  - None. Direct identifiers such as `Name`, `DOB`, `SSN`, `Zip` are present in the full dataset but are **not** included in `adult_small`.

- **Quasi-identifiers in `adult_small`**:
  - `Education`
  - `Marital Status`

  These are demographic attributes which, while not uniquely identifying by themselves, may be linkable to external data and thus enable re-identification when combined.

- **Sensitive attribute in `adult_small`**:
  - `Target` (income class, typically `<=50K` or `>50K`).

  We treat income category as sensitive information that should not be easily inferred, even if an attacker can identify the record’s equivalence class.

### 5. For which k ∈ [1, 10] does `adult_small` satisfy k-anonymity? (Task 1)

We consider k-anonymity with respect to quasi-identifiers:

```text
qis = ["Education", "Marital Status"]
```

Result:

- For **k = 1**:
  - `adult_small` **does satisfy** 1-anonymity.
  - Reason: every individual record belongs to some equivalence class of size at least 1 (trivially true for any non-empty dataset).

- For **k = 2, 3, ..., 10**:
  - `adult_small` **does not satisfy** k-anonymity.
  - Reason: there exists at least one combination of (`Education`, `Marital Status`) that appears **only once** among the first 100 rows. This smallest equivalence class has size 1, which is `< k` for any `k > 1`. Therefore, the k-anonymity condition fails for all k in `{2, 3, ..., 10}`.

### 6. Generalization and suppression for k = 2 (Task 2)

In Task 2, we assume that **`Target` is not a quasi-identifier**. We generalize categorical attributes and then suppress rows (delete records) to achieve 2-anonymity.

- **Generalization rules applied to `adult_small`**:
  - `Education` is generalized to two categories:
    - `< HS`: all education levels below `HS-grad` (e.g. `Preschool`, `1st-4th`, ..., `12th`).
    - `>= HS`: `HS-grad` and all higher education levels (e.g. `Some-college`, `Bachelors`, `Masters`, `Doctorate`, etc.).
  - `Marital Status` is generalized to:
    - `Married`: `Married-civ-spouse`, `Married-spouse-absent`, `Married-AF-spouse`.
    - `Not Married`: any other marital status (e.g. `Never-married`, `Divorced`, `Separated`, `Widowed`, etc.).

- **Suppression rule**:
  - After generalization, we compute equivalence classes based on the generalized quasi-identifiers:

    ```text
    qis = ["Education", "Marital Status"]
    ```

  - Any equivalence class whose size is **less than 2** is removed by deleting all rows in that class.
  - The resulting dataset is therefore **2-anonymous** with respect to the generalized `Education` and `Marital Status`.

### 7. For which rows is a homogeneity attack possible, and why? (Task 2)

After applying `generalize_categorical()` and suppression for `k = 2`:

- Each remaining record belongs to an equivalence class defined by the **generalized** quasi-identifiers:

```text
Education ∈ { "< HS", ">= HS" }
Marital Status ∈ { "Married", "Not Married" }
```

- A **homogeneity attack is possible** for those rows whose equivalence class has **only one distinct `Target` value**.

Concretely:

- Suppose, for some generalized group (for example, `Education = ">= HS"`, `Marital Status = "Married"`), every remaining record in that group has `Target = "<=50K"`. Then:
  - An attacker who knows that an individual belongs to this group (for example, knows they are married and have at least high-school education) can infer with **certainty** that this individual’s income class is `<=50K`.
  - Even though the group has at least 2 members (satisfies 2-anonymity), the **sensitive attribute is homogeneous** in this equivalence class.

The same reasoning applies to any group where all records share `Target = ">50K"`.

Therefore:

- **Rows vulnerable to homogeneity attack**:
  - All rows that belong to an equivalence class (defined by generalized `Education` and `Marital Status`) in which `Target` is **constant** (only one unique value).

- **Why**:
  - k-anonymity protects against identity disclosure but **does not** guarantee diversity of sensitive attributes within a group.
  - When a k-anonymous equivalence class is homogeneous in the sensitive attribute, knowing that someone is in that class reveals their sensitive attribute value.

