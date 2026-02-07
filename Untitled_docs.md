# Documentation: Untitled
> Auto-generated documentation for Jupyter notebook cells.
> Generated on: 2026-02-08 00:37:24

### Explanation
### High-Level Summary
This Python code defines a function `calculate_compound_interest` that computes the compound interest for a given principal amount, annual interest rate, and investment duration in years. The function returns a dictionary (`history`) containing the year-by-year growth of the investment, with each key representing the year and the corresponding value representing the accumulated amount at the end of that year. The code also includes a test case that calculates the compound interest for a $1000 principal at a 5% annual rate over 5 years, printing the final amount and the yearly breakdown.

### Detailed Breakdown
1. **Function Parameters**:
   - `principal` (float): The initial amount of money invested.
   - `rate` (float): The annual interest rate (expressed as a percentage, e.g., 5 for 5%).
   - `years` (int): The number of years the money is invested.

2. **Key Operations**:
   - The function initializes an empty dictionary `history` to store the yearly amounts and a variable `current_amount` set to the `principal`.
   - A loop iterates over each year from 1 to `years` (inclusive). For each year:
     - The interest earned is calculated as `current_amount * (rate / 100)`, converting the percentage rate to a decimal.
     - The interest is added to `current_amount`, updating the total investment value.
     - The updated `current_amount` is rounded to 2 decimal places (for currency precision) and stored in `history` with the year as the key.
   - The function returns the `history` dictionary, which contains the accumulated amount for each year.

3. **Test Case**:
   - The function is called with `principal=1000`, `rate=5`, and `years=5`.
   - The final amount after 5 years is printed (`result[5]`), and the entire yearly breakdown (`result`) is displayed.

### Relation to Previous Context
Since there is no previous context, this code stands alone as a self-contained utility for calculating compound interest. It could be part of a larger financial application or used independently for educational or practical purposes. The function is modular and reusable, making it easy to integrate into other projects requiring compound interest calculations.

```python
import math

def calculate_compound_interest(principal: float, rate: float, years: int) -> dict:
    """
    Calculates compound interest and returns a yearly breakdown.
    """
    history = {}
    current_amount = principal
    
    for year in range(1, years + 1):
        interest = current_amount * (rate / 100)
        current_amount += interest
        history[year] = round(current_amount, 2)
        
    return history

# Test with $1000 at 5% for 5 years
result = calculate_compound_interest(1000, 5, 5)
print(f"Final Amount: ${result[5]}")
print("Yearly Breakdown:", result)
```
