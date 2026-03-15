## Brief Run Manual

1. Open a terminal in the project folder:
   - `C:/Users/User/Desktop/privacy`

2. Run the main script:
   - `python e1.py`

3. Run tests:
   - `pytest e1.py`

## Exercise 1

Run for Exercise 1:

- `python e1.py`
- This prints the `adult_small` preview and k-anonymity results for `k = 1..10`.

• Which columns of adult_small are identifiers, which are quasi-identifiers, and which are sensitive attributes?

- `adult_small` columns: `Education`, `Marital Status`, `Target`
- Identifiers: none
- Quasi-identifiers: `Education`, `Marital Status`
- Sensitive attribute: `Target`

• For which k ∈ [1, 10] does adult_small satisfy k-anonymity? For the cases that do not
satisfy k-anonymity, why not?
- Satisfies k-anonymity for: `k = 1`
- Does not satisfy k-anonymity for: `k = 2, 3, 4, 5, 6, 7, 8, 9, 10`
- Why not for `k > 1`: at least one equivalence class defined by (`Education`, `Marital Status`)
  has size `1`, so the minimum class size is smaller than any `k > 1`.
The combination that appears only once for example is Bachelors,Separated 

## Exercise 2

Run test for Exercise 2:

- `python -m pytest e1.py -k generalize_categorical -q`

• For which rows is a homogeneity attack possible, and why?

- A homogeneity attack is possible for rows that belong to an equivalence class (after
  generalization and suppression for `k = 2`) where all records have the same `Target` value.
- In those classes, knowing a person's generalized quasi-identifiers (`Education`, `Marital Status`) reveals their sensitive value (`Target`) with certainty.
- This happens because k-anonymity protects identity by group size, but does not guarantee diversity of the sensitive attribute inside each group.
< HS,Married is an example where the target and the quasi identifier (after generalization and suppresion) are the same so if the attacker knows the group of the person he can directly figure out the target because there isnt another option in target

## Exercise 3

Run test for Exercise 3:

- `python -m pytest e1.py -k generalize_numeric -q`

## Exercise 4

Run test for Exercise 4:

- `python -m pytest e1.py -k make_adult_k_anonymous -q`

• How many rows would we have to suppress in addition to the generalization to achieve
k = 3 and k = 7?

Using QIs `["Zip", "Sex", "Age"]` and a practical setting `zip_digits = 2`, `age_digits = 2`:

- For `k = 3`: suppress **1** row.
- For `k = 7`: suppress **488** rows.

• How many digits of Zip and Age can/should you generalize, and how does this affect the
number of suppressed rows?

- Less generalization -> more suppression.
- More generalization -> fewer suppressed rows, but lower data utility.
- Example from the computed results:
  - `zip_digits = 2`, `age_digits = 2` -> suppression is low (`1` for `k=3`, `488` for `k=7`).
  - `zip_digits = 3`, `age_digits = 2` -> suppression is `0` for both `k=3` and `k=7`, but with stronger information loss.

## Exercise 5

Run test for Exercise 5:

- `python -m pytest e1.py -k l_diverse -q`

## Exercise 6

Assume that the quasi-identifiers in the adult dataset are `Education`, `Marital Status`, and
`Sex`, and that the sensitive column is `DOB`.

• Is the adult dataset `l-diverse?

- For `l = 2`, the adult dataset is **not** 2-diverse under both variants:
  - probabilistic 2-diversity: **False**
  - entropy 2-diversity: **False**

• If not, which quasi-identifier groups are preventing `l-diversity?

- The violating groups are QI groups where `DOB` is too concentrated (often only one row).
- In this dataset, the violating groups are singleton groups (size 1), for example:
  - `("Assoc-voc", "Married-AF-spouse", "Male")`
  - `("Preschool", "Divorced", "Male")`
  - `("Preschool", "Separated", "Female")`
  - `("Preschool", "Widowed", "Male")`
  - `("Prof-school", "Married-spouse-absent", "Female")`
  - `("Some-college", "Married-AF-spouse", "Male")`
- For each such group, one `DOB` value has probability 1 and entropy 0, so both 2-diversity conditions fail.

• What is the difference between probabilistic and entropy `l-diversity, if any?

- **Probabilistic l-diversity** checks only the largest sensitive-value probability in each group:
  `max_i p_i <= 1/l`.
- **Entropy l-diversity** uses the full sensitive-value distribution via entropy:
  `H(group) >= ln(l)`.
- In this dataset for `l = 2`, both give the same conclusion (not 2-diverse) because singleton
  groups force `p_max = 1` and `H = 0`.

## Exercise 7

Run test for Exercise 7:

- `python -m pytest e1.py -k "max_l or generalize_full_adult_categorical" -q`

• What is the largest ` for the generalized dataset for probabilistic `l-diversity and entropy
`l-diversity?

Using the generalized full dataset (Question 2 rules) with QIs
`["Education", "Marital Status", "Sex"]` and sensitive column `DOB`:

- Largest `l` for probabilistic l-diversity: **97**
- Largest `l` for entropy l-diversity: **193**

• Discuss the difference between probabilistic `l-diversity and entropy `l-diversity. Why does one of them appear more “diverse” than the other?

- Probabilistic l-diversity is controlled by only the most frequent sensitive value (`p_max`).
- Entropy l-diversity uses the whole sensitive-value distribution, not just the maximum.
- Because entropy includes the full distribution, it can rate a group as more diverse even when
  one value is somewhat dominant, as long as many other values still contribute to entropy.
- That is why entropy gives a higher maximum `l` here (**193**) than probabilistic (**97**).
