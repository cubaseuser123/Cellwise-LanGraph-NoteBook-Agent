# Documentation: Test
> Auto-generated documentation for Jupyter notebook cells.
> Generated on: 2026-02-08 00:55:36

### Explanation
### High-Level Summary
This Python code defines a function `calculate_compound_interest` that computes the compound interest for a given principal amount over a specified number of years at a fixed annual interest rate. The function returns a dictionary containing the yearly breakdown of the accumulated amount, rounded to two decimal places. The code also includes a test case that calculates the compound interest for a $1000 principal at a 5% annual rate over 5 years, printing the final amount and the yearly breakdown.

### Detailed Breakdown
1. **Function Parameters and Return Value**:
   - **Parameters**:
     - `principal` (float): The initial amount of money.
     - `rate` (float): The annual interest rate (as a percentage, e.g., 5 for 5%).
     - `years` (int): The number of years over which the interest is compounded.
   - **Return Value**: A dictionary (`history`) where the keys are the years (1 to `years`), and the values are the accumulated amounts at the end of each year, rounded to two decimal places.

2. **Key Operations**:
   - The function initializes an empty dictionary `history` to store the yearly amounts and a variable `current_amount` set to the `principal`.
   - A loop iterates over each year from 1 to `years`. For each year:
     - The interest for the year is calculated as `current_amount * (rate / 100)`.
     - The interest is added to `current_amount`, compounding the amount.
     - The updated `current_amount` is rounded to two decimal places and stored in the `history` dictionary with the year as the key.
   - The function returns the `history` dictionary after processing all years.

3. **Test Case**:
   - The function is tested with `principal = 1000`, `rate = 5`, and `years = 5`.
   - The final amount after 5 years is printed (`result[5]`), and the entire yearly breakdown (`result`) is displayed.

### Relation to Previous Context
Since there is no previous context, this code stands alone as a self-contained implementation for calculating compound interest. It could be part of a larger financial application or used as a utility function for interest calculations. The modular design allows it to be easily integrated into other scripts or applications requiring compound interest computations.

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
### Explanation
### High-Level Summary
This Python code defines a function `calculate_compound_interest` that computes the compound interest for a given principal amount over a specified number of years at a fixed annual interest rate. The function returns a dictionary (`history`) where each key is a year, and the corresponding value is the accumulated amount at the end of that year, rounded to two decimal places. The code also includes a test case that calculates the compound interest for a $1000 principal at a 5% annual rate over 5 years, printing both the final amount and the yearly breakdown.

### Detailed Breakdown
1. **Function Parameters**:
   - `principal` (float): The initial amount of money.
   - `rate` (float): The annual interest rate (expressed as a percentage, e.g., 5 for 5%).
   - `years` (int): The number of years over which the interest is compounded.

2. **Key Operations**:
   - The function initializes an empty dictionary `history` to store the yearly amounts and a variable `current_amount` set to the `principal`.
   - A `for` loop iterates over each year from 1 to `years` (inclusive). For each year:
     - The interest for the year is calculated as `current_amount * (rate / 100)`. This converts the percentage rate to a decimal (e.g., 5% becomes 0.05) and multiplies it by the current amount.
     - The interest is added to `current_amount`, updating it for the next iteration.
     - The updated `current_amount` is rounded to two decimal places and stored in the `history` dictionary with the year as the key.
   - The function returns the `history` dictionary, which contains the yearly breakdown of the compounded amount.

3. **Test Case**:
   - The function is called with `principal=1000`, `rate=5`, and `years=5`. The result is stored in `result`.
   - The final amount after 5 years (`result[5]`) is printed, followed by the entire yearly breakdown (`result`).

### Relation to Previous Context
Since there is no previous context, this code stands alone as a self-contained implementation for calculating compound interest. It could be part of a larger financial application or used as a utility function for interest calculations. The test case demonstrates its usage and verifies correctness for a simple scenario.

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
