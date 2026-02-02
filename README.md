# Decimal Digit Extraction for π via Machin's Formula

This repository contains an algorithm for extracting arbitrary decimal digits of π without computing preceding digits, based on Machin's classical formula.

## Contents

- `modular_bbp.py` - Python implementation of the algorithm
- `machin_digit_extract.cpp` - High-performance C++ implementation with OpenMP
- `pi_decimal_paper.tex` - LaTeX source for the paper
- `pi_decimal_paper.pdf` - Compiled paper

## Python Usage

```python
from modular_bbp import extract_decimal_digit

# Extract the 1000th decimal digit of π
digit = extract_decimal_digit(1000)
print(digit)  # Output: 9
```

## C++ Usage (faster)

Requires GMP, MPFR, and OpenMP.

```bash
# macOS with Homebrew
clang++ -std=c++17 -O3 -Xpreprocessor -fopenmp \
  -I/opt/homebrew/opt/libomp/include -I/opt/homebrew/include \
  -L/opt/homebrew/opt/libomp/lib -L/opt/homebrew/lib \
  -o machin_digit_extract machin_digit_extract.cpp -lmpfr -lgmp -lomp

# Linux
g++ -std=c++17 -O3 -fopenmp -o machin_digit_extract machin_digit_extract.cpp -lmpfr -lgmp

./machin_digit_extract
```

## Performance

- Python: ~39 seconds to verify first 1000 digits
- C++ (8 threads): ~5 seconds to verify first 1000 digits, position 10,000 in under 5 seconds

## Reference

J. Babcanec, "Decimal Digit Extraction for π via Machin's Formula: Series Splitting and Borrow-Tracking Methods"
