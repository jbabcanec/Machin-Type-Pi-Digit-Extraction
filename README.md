# Decimal Digit Extraction for π via Machin's Formula

This repository contains an algorithm for extracting arbitrary decimal digits of π without computing preceding digits, based on Machin's classical formula.

## Contents

- `modular_bbp.py` - Python implementation of the algorithm
- `pi_decimal_paper.tex` - LaTeX source for the paper
- `pi_decimal_paper.pdf` - Compiled paper

## Usage

```python
from modular_bbp import extract_decimal_digit

# Extract the 1000th decimal digit of π
digit = extract_decimal_digit(1000)
print(digit)  # Output: 9
```

## Verification

```bash
python modular_bbp.py
```

This runs verification against known digits up to position 50 and benchmarks at larger positions.

## Reference

J. Babcanec, "Decimal Digit Extraction for π via Machin's Formula: Series Splitting and Borrow-Tracking Methods"
