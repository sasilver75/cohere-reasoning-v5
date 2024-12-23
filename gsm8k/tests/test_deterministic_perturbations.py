import pytest
import sys
from pathlib import Path

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
