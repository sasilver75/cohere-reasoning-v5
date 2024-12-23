import pytest
import sys
from pathlib import Path
import re

# Add the parent directory to sys.path; I don't want to fight with imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from gsm8k.deterministic_perturbations import (
    NumericModification,
    OperatorSwapModification,
    UnitModification,
    PercentageModification,
    FractionModification,
    OrderOfOperationsModification,
    EquationBalanceErrorModification,
    NothingModification
)

def test_numeric_modification():
    # Test applicability
    assert NumericModification.is_applicable("There are 5 apples")
    assert NumericModification.is_applicable("The price is 3.14")
    assert not NumericModification.is_applicable("There are no numbers here")

    # Test application
    text = "There are 10 apples"
    result = NumericModification.apply(text)
    assert result != text  # Should be modified
    assert "apples" in result  # Context should remain
    
    # Test decimal numbers
    text = "The price is 3.14"
    result = NumericModification.apply(text)
    assert result != text
    assert "The price is" in result

def test_numeric_modification_formatting():
    """Test that numeric modifications preserve appropriate number formatting"""
    
    # Test integer formatting
    text = "The cost is $130000"
    result = NumericModification.apply(text)
    # Should still be an integer, no decimal points
    assert '.' not in result.split('$')[1]
    
    # Test decimal precision preservation
    text = "The price is 3.14159"
    result = NumericModification.apply(text)
    # Should maintain 5 decimal places
    decimal_part = result.split('.')[-1]
    assert len(decimal_part) == 5
    
    # Test multiple numbers in text
    text = "First number 100, second number 3.14"
    result = NumericModification.apply(text)
    numbers = re.findall(r'\d+(?:\.\d+)?', result)
    for num in numbers:
        if '.' in num:
            # If it was a decimal, should have 2 decimal places
            assert len(num.split('.')[-1]) == 2
        else:
            # If it was an integer, should still be an integer
            assert '.' not in num
    
    # Test large numbers
    text = "The total is 1000000"
    result = NumericModification.apply(text)
    modified_num = re.search(r'\d+', result).group()
    # Should be an integer, no scientific notation
    assert 'e' not in modified_num.lower()
    assert '.' not in modified_num
    
    # Test zero decimal places
    text = "The temperature is 98.0 degrees"
    result = NumericModification.apply(text)
    modified_num = re.search(r'\d+\.\d+', result).group()
    # Should maintain one decimal place
    assert len(modified_num.split('.')[-1]) == 1

def test_numeric_modification_large_numbers():
    """Test that numeric modifications handle large numbers and commas correctly"""
    
    # Test large numbers with commas
    text = "The house costs $80,000"
    result = NumericModification.apply(text)
    # Should preserve comma formatting
    modified_num = re.search(r'\d+(?:,\d{3})*', result.split('$')[1]).group()
    assert ',' in modified_num
    assert len(modified_num.replace(',', '')) >= 3  # Should still be a reasonable number
    
    # Test multiple large numbers
    text = "First price $80,000, second price $120,000"
    result = NumericModification.apply(text)
    numbers = re.findall(r'\d+(?:,\d{3})*', result)
    for num in numbers:
        assert ',' in num  # All large numbers should have commas
    
    # Test mixed format numbers
    text = "Price range: $80,000.50 to $120,000.75"
    result = NumericModification.apply(text)
    numbers = re.findall(r'\d+(?:,\d{3})*\.\d+', result)
    for num in numbers:
        # Should preserve both commas and decimal places
        assert ',' in num and '.' in num
        whole, decimal = num.split('.')
        assert len(decimal) == 2  # Should preserve 2 decimal places

def test_operator_swap_modification():
    # Test applicability
    assert OperatorSwapModification.is_applicable("2 + 2 = 4")
    assert OperatorSwapModification.is_applicable("5 * 3 = 15")
    assert not OperatorSwapModification.is_applicable("There are no operators")

    # Test application
    text = "2 + 2 = 4"
    result = OperatorSwapModification.apply(text)
    assert "2 - 2" in result or "2 + 2" in result  # Should either change or keep operator

    text = "if x > y then"
    result = OperatorSwapModification.apply(text)
    assert "x < y" in result or "x > y" in result

def test_operator_swap_markdown_safety():
    """Test that operator swap doesn't affect Markdown bullet points"""
    
    # Test with Markdown bullet points
    text = """To solve this:
    - First add 2 + 3
    - Then multiply by 4
    - Finally subtract 1"""
    result = OperatorSwapModification.apply(text)
    assert "- First" in result  # Bullet point should remain
    assert "- Then" in result   # Bullet point should remain
    assert "- Finally" in result  # Bullet point should remain
    
    # Test with mathematical operators
    text = "2 + 3 = 5"
    result = OperatorSwapModification.apply(text)
    assert result != text  # Should modify actual mathematical operator
    assert "2 - 3 = 5" in result or "2 + 3 = 5" in result

    # Test with mixed content
    text = """Steps:
    - First calculate 2 + 3
    - Then calculate 5 - 2
    - Result is 3"""
    result = OperatorSwapModification.apply(text)
    assert "- First" in result  # Bullet points should remain
    assert "- Then" in result
    assert "- Result" in result
    # Should modify one of the mathematical operators but not bullet points
    assert (("2 - 3" in result) != ("2 + 3" in result)) or (("5 + 2" in result) != ("5 - 2" in result))

def test_operator_swap_specific_cases():
    """Test specific operator swap cases"""
    
    # Test minus between numbers
    text = "5 - 3 = 2"
    result = OperatorSwapModification.apply(text)
    assert "5 + 3" in result or "5 - 3" in result

    # Test plus between numbers
    text = "5 + 3 = 8"
    result = OperatorSwapModification.apply(text)
    assert "5 - 3" in result or "5 + 3" in result

    # Test with comment-like content
    text = "/* This is a comment with + and - */"
    result = OperatorSwapModification.apply(text)
    assert result == text  # Should not modify operators in comments

    # Test with equation in markdown list
    text = """Problem:
    - Given x + y = 10
    - And x - y = 2
    Solve for x and y"""
    result = OperatorSwapModification.apply(text)
    # Should modify one equation operator but not bullet points
    assert text.count("-") - result.count("-") in [-1, 0, 1]  # Only one - should change at most

def test_unit_modification():
    # Test applicability
    assert UnitModification.is_applicable("5 hours of work")
    assert UnitModification.is_applicable("10 kilometers away")
    assert not UnitModification.is_applicable("no units here")

    # Test application
    text = "5 hours of work"
    result = UnitModification.apply(text)
    assert "minutes" in result
    assert "5" in result  # Number should remain unchanged

    text = "10 kilometers to school"
    result = UnitModification.apply(text)
    assert "miles" in result
    assert "10" in result

def test_percentage_modification():
    # Test applicability
    assert PercentageModification.is_applicable("20% of students")
    assert PercentageModification.is_applicable("25 percent increase")
    assert not PercentageModification.is_applicable("no percentages here")

    # Test application
    text = "20% of students"
    result = PercentageModification.apply(text)
    assert result != text
    assert "%" in result or "percent" in result

    text = "25 percent increase"
    result = PercentageModification.apply(text)
    assert result != text
    assert "percent" in result

def test_fraction_modification():
    # Test applicability
    assert FractionModification.is_applicable("3/4 of the cake")
    assert not FractionModification.is_applicable("no fractions here")

    # Test application
    text = "3/4 of the cake"
    result = FractionModification.apply(text)
    assert result != text
    assert "/" in result
    assert "of the cake" in result

    # Test multiple fractions
    text = "1/2 + 3/4"
    result = FractionModification.apply(text)
    assert result != text
    assert "/" in result

def test_order_of_operations_modification():
    # Test applicability
    assert OrderOfOperationsModification.is_applicable("(2 + 3) * 4")
    assert not OrderOfOperationsModification.is_applicable("2 + 3 * 4")

    # Test application
    text = "(2 + 3) * 4"
    result = OrderOfOperationsModification.apply(text)
    assert result != text
    assert "2" in result and "3" in result and "4" in result

def test_equation_balance_error_modification():
    # Test applicability
    assert EquationBalanceErrorModification.is_applicable("2 + 2 = 4")
    assert EquationBalanceErrorModification.is_applicable("x = 5")
    assert not EquationBalanceErrorModification.is_applicable("no equations here")

    # Test application
    text = "2 + 2 = 4"
    result = EquationBalanceErrorModification.apply(text)
    assert result != text
    assert "2 + 2 =" in result  # Left side should remain unchanged

    text = "x = 5.5"
    result = EquationBalanceErrorModification.apply(text)
    assert result != text
    assert "x =" in result

def test_nothing_modification():
    # Test applicability (should always be applicable)
    assert NothingModification.is_applicable("any text")
    assert NothingModification.is_applicable("")

    # Test application (should return unchanged text)
    text = "This is a test"
    assert NothingModification.apply(text) == text

    text = ""
    assert NothingModification.apply(text) == text

@pytest.mark.parametrize("text,expected_applicable", [
    ("5 apples", True),  # NumericModification
    ("2 + 3", True),     # OperatorSwapModification
    ("5 hours", True),   # UnitModification
    ("20%", True),       # PercentageModification
    ("3/4", True),       # FractionModification
    ("(2 + 3)", True),   # OrderOfOperationsModification
    ("x = 5", True),     # EquationBalanceErrorModification
    ("no numbers", True), # NothingModification (always applicable)
])
def test_at_least_one_strategy_applicable(text, expected_applicable):
    """Test that at least one strategy is applicable for relevant texts"""
    strategies = [
        NumericModification,
        OperatorSwapModification,
        UnitModification,
        PercentageModification,
        FractionModification,
        OrderOfOperationsModification,
        EquationBalanceErrorModification,
        NothingModification
    ]
    
    assert any(s.is_applicable(text) for s in strategies) == expected_applicable

def test_order_of_operations_strict_matching():
    """Test that OrderOfOperationsModification only matches simple arithmetic expressions"""
    
    # Should match these simple cases
    assert OrderOfOperationsModification.is_applicable("(2 + 3) * 4")
    assert OrderOfOperationsModification.is_applicable("(5 - 1) / 2")
    assert OrderOfOperationsModification.is_applicable("(10 + 20) * 3")
    assert OrderOfOperationsModification.is_applicable("2 + (3 * 4)")  # Different parentheses position
    assert OrderOfOperationsModification.is_applicable("5 * (2 + 1)")  # Different parentheses position
    
    # Should NOT match these cases
    assert not OrderOfOperationsModification.is_applicable(r"Total is \( 0.4 \times 200 = 80 \)")  # LaTeX
    assert not OrderOfOperationsModification.is_applicable("First (add 2 and 3) then multiply")  # Text description
    assert not OrderOfOperationsModification.is_applicable("(x + y) * z")  # Variables
    assert not OrderOfOperationsModification.is_applicable("(2 + 3 + 4) * 5")  # More than two numbers
    
    # Test application on valid cases with parentheses at start
    text = "(2 + 3) * 4"
    result = OrderOfOperationsModification.apply(text)
    assert result in [
        "2 + 3 * 4",  # Removed parentheses
        "2 * (3 * 4)"  # Shifted parentheses
    ]
    
    # Test application on valid cases with parentheses in middle
    text = "2 + (3 * 4)"
    result = OrderOfOperationsModification.apply(text)
    assert result in [
        "2 + 3 * 4",  # Removed parentheses
        "(2 + 3) * 4"  # Shifted parentheses
    ]
    
    # Test that complex expressions are preserved
    text = r"The formula is \( (a + b) * c \) in math notation"
    assert OrderOfOperationsModification.apply(text) == text
    
    # Test mixed content
    text = """Here's the calculation:
    (5 + 3) * 2 = 16
    And also \( (x + y) * z \) in LaTeX"""
    result = OrderOfOperationsModification.apply(text)
    # Should only modify the simple arithmetic expression
    assert r"\( (x + y) * z \)" in result  # LaTeX preserved
    assert result != text  # But something was modified

def test_order_of_operations_pattern_consistency():
    """Test that BASE_PATTERNS and CAPTURE_PATTERNS match the same expressions"""
    
    test_cases = [
        # Should match both pattern sets
        "(2 + 3) * 4",
        "(5 - 1) / 2",
        "2 + (3 * 4)",
        "5 * (2 + 1)",
        "(10 + 20) * 3",
        "4 + (6 * 2)",
        
        # Should match neither pattern set
        r"Total is \( 0.4 \times 200 = 80 \)",  # LaTeX
        "First (add 2 and 3) then multiply",    # Text description
        "(x + y) * z",                          # Variables
        "(2 + 3 + 4) * 5",                      # More than two numbers
        "2 * 3 + 4",                            # No parentheses
    ]
    
    for text in test_cases:
        # Check if base patterns match
        base_matches = any(bool(re.search(pattern, text)) 
                         for pattern in OrderOfOperationsModification.BASE_PATTERNS)
        
        # Check if capture patterns match
        capture_matches = any(bool(re.search(pattern, text)) 
                            for pattern, _ in OrderOfOperationsModification.CAPTURE_PATTERNS)
        
        # Both pattern sets should agree
        assert base_matches == capture_matches, \
            f"Pattern mismatch for '{text}': base_matches={base_matches}, capture_matches={capture_matches}"
        
        # If applicable, verify that apply() actually modifies the text
        if base_matches:
            result = OrderOfOperationsModification.apply(text)
            assert result != text, f"apply() didn't modify applicable text: '{text}'"
        else:
            result = OrderOfOperationsModification.apply(text)
            assert result == text, f"apply() modified non-applicable text: '{text}'"

def test_order_of_operations_transformations():
    """Test specific transformations of the OrderOfOperationsModification"""
    
    # Test front parentheses transformations
    text = "(2 + 3) * 4"
    result = OrderOfOperationsModification.apply(text)
    assert result in [
        "2 + 3 * 4",      # Removed parentheses
        "2 * (3 * 4)"     # Shifted parentheses
    ]
    
    # Test middle parentheses transformations
    text = "2 + (3 * 4)"
    result = OrderOfOperationsModification.apply(text)
    assert result in [
        "2 + 3 * 4",      # Removed parentheses
        "(2 + 3) * 4"     # Shifted parentheses
    ]

def test_numeric_modification_edge_cases():
    """Test numeric modifications with various edge cases"""
    
    # Test trailing zeros in decimals
    text = "Temperature is 98.00 degrees"
    result = NumericModification.apply(text)
    modified_num = re.search(r'\d+\.\d+', result).group()
    assert len(modified_num.split('.')[1]) == 2, "Should preserve trailing zeros"
    
    # Test very large numbers with commas
    text = "The budget is $1,234,567,890"
    result = NumericModification.apply(text)
    modified_num = re.search(r'\d+(?:,\d{3})*', result.split('$')[1]).group()
    # Verify comma placement is correct (every 3 digits from right)
    digits = modified_num.replace(',', '')
    expected_commas = (len(digits) - 1) // 3  # Number of commas there should be
    assert modified_num.count(',') == expected_commas, f"Expected {expected_commas} commas in {modified_num}"
    # Verify each group is exactly 3 digits (except possibly first group)
    groups = modified_num.split(',')
    assert all(len(g) == 3 for g in groups[1:]), f"All groups after first should be length 3: {groups}"
    assert 1 <= len(groups[0]) <= 3, f"First group should be 1-3 digits, got {len(groups[0])}"
    
    # Test mixed formats in same text
    text = "Cost ranges from $1,234.50 to $5,678.00"
    result = NumericModification.apply(text)
    numbers = re.findall(r'\d+(?:\.\d+)?', result)  # Find all numbers first
    for num in numbers:
        # Only check for commas if number is large enough
        if len(num.split('.')[0]) > 3:  # More than 3 digits before decimal
            num_with_commas = re.search(r'\d+(?:,\d{3})*(?:\.\d+)?', result).group()
            assert ',' in num_with_commas, f"Large number {num} should have commas"
        if '.' in num:
            assert len(num.split('.')[1]) == 2, f"Number {num} should have 2 decimal places"
    
    # Test negative numbers
    text = "Temperature dropped to -15.5 degrees"
    result = NumericModification.apply(text)
    modified_num = re.search(r'-?\d+\.\d+', result).group()
    assert len(modified_num.split('.')[1]) == 1, "Should preserve single decimal place"
    assert modified_num.startswith('-') or float(modified_num) < 0, "Should preserve negative value"
    
    # Test currency format preservation
    text = "Price: $1,234.99"
    result = NumericModification.apply(text)
    # First verify we can find a number
    modified_num = re.search(r'\d+(?:\.\d+)?', result)
    assert modified_num is not None, "Could not find modified number in result"
    modified_num = modified_num.group()
    
    # If original had decimal places, modified version should maintain 2 decimal places
    if '.' in modified_num:
        whole_part, decimal_part = modified_num.split('.')
        assert len(decimal_part) == 2, f"Currency decimal places should be 2, got {len(decimal_part)}"
    
    # Check comma formatting if number is large enough (≥1000)
    whole_part = modified_num.split('.')[0] if '.' in modified_num else modified_num
    if len(whole_part) >= 4:  # Only expect commas for numbers >= 1000
        num_with_commas = re.search(r'\d+(?:,\d{3})*(?:\.\d+)?', result)
        assert num_with_commas is not None, f"Large number {modified_num} should have commas"
        assert ',' in num_with_commas.group(), f"Large number {modified_num} should have commas"

    # Test additional currency cases
    for test_case in ["$5", "$1,234", "$1.99", "$999.99", "$1,000.00"]:
        result = NumericModification.apply(test_case)
        # Use a pattern that can handle commas in the number
        modified_num = re.search(r'\d+(?:,\d{3})*(?:\.\d+)?', result.split('$')[1]).group()

        # If original had decimals, keep 2 decimal places
        if '.' in test_case:
            assert '.' in modified_num, f"Should preserve decimal format from {test_case}"
            assert len(modified_num.split('.')[1]) == 2, f"Should keep 2 decimal places from {test_case}"
        
        # If number is ≥1000, should have commas
        whole_part = modified_num.split('.')[0] if '.' in modified_num else modified_num
        if len(whole_part) >= 4:
            assert ',' in modified_num, f"Number >= 1000 ({modified_num}) should have commas"

def test_numeric_modification_consistency():
    """Test that numeric modifications are consistent in their formatting"""
    
    # Run multiple times to ensure consistency despite random choices
    for _ in range(10):
        # Test decimal places preservation
        text = "Value is 123.4500"
        result = NumericModification.apply(text)
        modified_num = re.search(r'\d+\.\d+', result).group()
        assert len(modified_num.split('.')[1]) == 4  # Should always keep 4 decimal places
        
        # Test comma formatting preservation
        text = "Cost is $1,234,567"
        result = NumericModification.apply(text)
        modified_num = re.search(r'\d+(?:,\d{3})*', result.split('$')[1]).group()
        assert ',' in modified_num  # Should always have commas
        assert '.' not in modified_num  # Should never add decimals
