from abc import ABC, abstractmethod
import re
import random
from typing import Optional, Tuple

class PerturbationStrategy(ABC):
    """
    Base ABC class for all deterministic perturbation strategies.
    """
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.NAME is None:
            raise TypeError(f"Can't instantiate abstract class {cls.__name__} without NAME defined")
    
    NAME: str = None  # Must be assigned in subclasses
    
    @staticmethod
    @abstractmethod
    def is_applicable(text: str) -> bool:
        """Check if this perturbation can be applied to the text"""
        pass

    @staticmethod
    @abstractmethod
    def apply(text: str) -> str:
        """Apply the perturbation to the text"""
        pass

class NumericModification(PerturbationStrategy):
    """
    Modifies standalone numeric values in text.
    
    Applicability:
        - Text contains any number (integer or decimal)
        - Examples: "5", "3.14", "100", "$1,234.99"
    
    Perturbation:
        Applies one of several transformations while preserving format:
        - Increase by 50% (multiply by 1.5)
        - Decrease by 50% (multiply by 0.5)
        - Increase by order of magnitude (multiply by 10)
        - Decrease by order of magnitude (multiply by 0.1)
        - Change sign (multiply by -1)
        
        Format preservation:
        - Maintains comma grouping for large numbers
        - Preserves decimal precision
        - Always shows 2 decimal places for currency values
    """
    NAME = "NumericModification"

    @staticmethod
    def format_with_commas(number_str: str) -> str:
        """Helper to add commas to a number string"""
        parts = []
        while number_str:
            parts.append(number_str[-3:])
            number_str = number_str[:-3]
        return ','.join(reversed(parts))

    @staticmethod
    def is_applicable(text: str) -> bool:
        return bool(re.search(r'\d+(?:,\d{3})*(?:\.\d+)?', text))
    
    @staticmethod
    def apply(text: str) -> str:
        # Include optional $ in the pattern, but in a non-capturing group
        numbers = list(re.finditer(r'(?P<currency>\$)?(?P<number>\d+(?:,\d{3})*(?:\.\d+)?)', text))
        if not numbers:
            return text
        
        target = random.choice(numbers)
        original_text = target.group('number')  # Get just the number part
        is_currency = bool(target.group('currency'))  # Check if we matched a $ sign
        original_had_decimal = '.' in target.group()  # Check if original had decimal
        
        # Determine format from original text
        had_commas = ',' in original_text
        had_decimal = '.' in original_text
        decimal_places = len(original_text.split('.')[1]) if had_decimal else 0
        
        # Convert and transform
        num_str = original_text.replace(',', '')
        num = float(num_str)
        new_num = random.choice([
            lambda x: x * 1.5,
            lambda x: x * 0.5,
            lambda x: x * 10,
            lambda x: x * 0.1,
            lambda x: -x,
        ])(num)
        
        # Format according to original format
        whole_part = str(abs(int(new_num)))
        
        # Add commas if original had them or if it's a large number in currency format
        needs_commas = (had_commas or (is_currency and len(whole_part) >= 4))
        if needs_commas:
            whole_part = NumericModification.format_with_commas(whole_part)
        
        # Add decimal places if original had them
        if original_had_decimal:  # Use original_had_decimal here
            decimal_part = abs(new_num) % 1
            decimal_str = str(decimal_part)[2:].ljust(2 if is_currency else decimal_places, '0')
            decimal_str = decimal_str[:2 if is_currency else decimal_places]
            formatted_num = f"{'-' if new_num < 0 else ''}{whole_part}.{decimal_str}"
        else:
            formatted_num = f"{'-' if new_num < 0 else ''}{whole_part}"
        
        # Add back the $ if it was present
        if is_currency:
            formatted_num = f"${formatted_num}"
        
        return text[:target.start()] + formatted_num + text[target.end():]

class OperatorSwapModification(PerturbationStrategy):
    """
    Swaps mathematical operators with their opposites.
    """
    NAME = "OperatorSwapModification"

    # Define regex patterns for matching operators in mathematical contexts
    OPERATOR_PATTERNS = [
        # Match + or - between numbers with optional whitespace
        (r'(?<=\d)\s*\+\s*(?=\d)', r'-'),  # + between numbers
        (r'(?<=\d)\s*-\s*(?=\d)', r'+'),  # - between numbers
        # Match * and / between numbers
        (r'(?<=\d)\s*\*\s*(?=\d)', r'/'),  # * between numbers
        (r'(?<=\d)\s*/\s*(?=\d)', r'*'),  # / between numbers
        # Match comparison operators between numbers
        (r'(?<=\d)\s*>\s*(?=\d)', r'<'),  # > between numbers
        (r'(?<=\d)\s*<\s*(?=\d)', r'>'),  # < between numbers
        # Match compound comparison operators
        (r'(?<=\d)\s*>=\s*(?=\d)', r'<='),
        (r'(?<=\d)\s*<=\s*(?=\d)', r'>=')
    ]
    
    @staticmethod
    def is_applicable(text: str) -> bool:
        # Check if any pattern matches in a mathematical context
        return any(re.search(pattern[0], text) 
                  for pattern in OperatorSwapModification.OPERATOR_PATTERNS)
    
    @staticmethod
    def apply(text: str) -> str:
        # Find all applicable patterns
        applicable_pairs = [
            pair for pair in OperatorSwapModification.OPERATOR_PATTERNS 
            if re.search(pair[0], text)
        ]
        if not applicable_pairs:
            return text
            
        # Choose and apply a random pattern
        op_pair = random.choice(applicable_pairs)
        # print(f"Found applicable operator pair: {op_pair}")  # Debug print
        # print(f"Matches found: {re.findall(op_pair[0], text)}")  # Debug print
        
        # Apply the substitution
        result = re.sub(op_pair[0], op_pair[1], text, count=1)
        # print(f"Text before: {text}")  # Debug print
        # print(f"Text after: {result}")  # Debug print
        return result

class UnitModification(PerturbationStrategy):
    """
    Changes units of measurement without converting values.
    
    Applicability:
        - Text contains a number followed by a unit of measurement
        - Supported units: hours/minutes, days/weeks, meters/feet,
          kilometers/miles, kilograms/pounds, dollars/euros
    
    Perturbation:
        Replaces one unit with its counterpart without adjusting the number:
        - "5 hours" → "5 minutes"
        - "10 kilometers" → "10 miles"
        This creates errors because the values aren't converted appropriately.
    """
    NAME = "UnitModification"

    UNIT_PAIRS = [
        (r'hours?', 'minutes'),
        (r'days?', 'weeks'),
        (r'meters?', 'feet'),
        (r'kilometers?', 'miles'),
        (r'kilograms?', 'pounds'),
        (r'dollars?', 'euros'),
    ]
    
    @staticmethod
    def is_applicable(text: str) -> bool:
        return any(re.search(rf'\d+\s*{unit[0]}', text, re.IGNORECASE) 
                  for unit in UnitModification.UNIT_PAIRS)
    
    @staticmethod
    def apply(text: str) -> str:
        applicable_pairs = [
            pair for pair in UnitModification.UNIT_PAIRS 
            if re.search(rf'\d+\s*{pair[0]}', text, re.IGNORECASE)
        ]
        pair = random.choice(applicable_pairs)
        return re.sub(pair[0], pair[1], text, count=1)

class PercentageModification(PerturbationStrategy):
    """
    Modifies percentage values or their interpretation.
    
    Applicability:
        - Text contains a number followed by % or the word "percent"
        - Example: "20%" or "20 percent"
    
    Perturbation:
        Applies one of several transformations:
        - Takes complement (e.g., 20% → 80%)
        - Doubles the percentage
        - Halves the percentage
        - Changes "X% of" to "X% more than" (changes meaning significantly)
    """
    NAME = "PercentageModification"

    @staticmethod
    def is_applicable(text: str) -> bool:
        return bool(re.search(r'\d+(?:\s)?(?:%|percent)', text))
    
    @staticmethod
    def apply(text: str) -> str:
        matches = list(re.finditer(r'(\d+)(?:\s)?(%|percent)', text))
        if not matches:
            return text
        
        match = random.choice(matches)
        num = int(match.group(1))
        
        strategies = [
            lambda x: 100 - x,  # Complement (e.g., 20% → 80%)
            lambda x: x * 2,    # Double
            lambda x: x / 2,    # Half
            # Convert "X% of" to "X% more than"
            lambda x: f"{x}% more than" if "% of" in text[match.start()-3:match.end()+3] else f"{x}%"
        ]
        
        new_value = random.choice(strategies)(num)
        return text[:match.start()] + str(new_value) + match.group(2) + text[match.end():]

class FractionModification(PerturbationStrategy):
    """
    Modifies fractions by changing numerator or denominator.
    
    Applicability:
        - Text contains a fraction in the format "X/Y"
        - Example: "3/4", "1/2"
    
    Perturbation:
        Applies one of several transformations:
        - Inverts the fraction (3/4 → 4/3)
        - Doubles numerator
        - Doubles denominator
    """
    NAME = "FractionModification"

    @staticmethod
    def is_applicable(text: str) -> bool:
        return bool(re.search(r'\d+/\d+', text))
    
    @staticmethod
    def apply(text: str) -> str:
        matches = list(re.finditer(r'(\d+)/(\d+)', text))
        if not matches:
            return text
            
        match = random.choice(matches)
        num, denom = int(match.group(1)), int(match.group(2))
        
        strategies = [
            lambda n, d: (d, n),  # Invert fraction
            lambda n, d: (n*2, d),  # Double numerator
            lambda n, d: (n, d*2),  # Double denominator
        ]
        
        new_num, new_denom = random.choice(strategies)(num, denom)
        return text[:match.start()] + f"{new_num}/{new_denom}" + text[match.end():]

class OrderOfOperationsModification(PerturbationStrategy):
    """
    Modifies simple mathematical expressions by changing parentheses placement.
    
    Applicability:
        - Text contains SIMPLE arithmetic expressions with parentheses
        - ONLY matches patterns like:
          - "(2 + 3) * 4"
          - "(5 - 1) / 2"
          - "2 + (3 * 4)"
          - "5 * (2 + 1)"
        - Does NOT match:
          - LaTeX equations \( ... \)
          - Complex expressions
          - Text descriptions of math
    """
    NAME = "OrderOfOperationsModification"

    # Define the basic patterns without capture groups
    BASE_PATTERNS = [
        r'\(\d+\s*[+\-*/]\s*\d+\)\s*[+\-*/]\s*\d+',  # (2 + 3) * 4
        r'\d+\s*[+\-*/]\s*\(\d+\s*[+\-*/]\s*\d+\)'   # 2 + (3 * 4)
    ]

    # Define the patterns with capture groups for extraction
    CAPTURE_PATTERNS = [
        (r'(\(\d+\s*[+\-*/]\s*\d+\))\s*([+\-*/])\s*(\d+)',  # (2 + 3) * 4
         lambda m: [
             f"{m.group(1)[1:-1]} {m.group(2)} {m.group(3)}",  # Remove parentheses
             f"{m.group(1).split()[0][1:]} {m.group(2)} ({m.group(1).split()[-1][:-1]} {m.group(2)} {m.group(3)})"  # Shift parentheses
         ]),
        (r'(\d+)\s*([+\-*/])\s*(\(\d+\s*[+\-*/]\s*\d+\))',  # 2 + (3 * 4)
         lambda m: [
             f"{m.group(1)} {m.group(2)} {m.group(3)[1:-1]}",  # Remove parentheses
             f"({m.group(1)} {m.group(2)} {m.group(3).split()[0][1:]}) {m.group(3).split()[1]} {m.group(3).split()[-1][:-1]}"  # Shift parentheses
         ])
    ]

    @staticmethod
    def is_applicable(text: str) -> bool:
        return any(bool(re.search(pattern, text)) for pattern in OrderOfOperationsModification.BASE_PATTERNS)
    
    @staticmethod
    def apply(text: str) -> str:
        for pattern, strategy_generator in OrderOfOperationsModification.CAPTURE_PATTERNS:
            matches = list(re.finditer(pattern, text))
            if matches:
                match = random.choice(matches)
                strategies = strategy_generator(match)
                new_expr = random.choice(strategies)
                return text[:match.start()] + new_expr + text[match.end():]
        return text
    

# class EquationBalanceErrorModification(PerturbationStrategy):
#     """
#     Introduces errors in equation results while preserving the calculation.
    
#     Applicability:
#         - Text contains an equation where the right side is a number
#         - Example: "2 + 2 = 4", "3 * 5 = 15", "60 × 3 = 180"
    
#     Perturbation:
#         Modifies only the result (right side) while keeping the calculation:
#         - Adds or subtracts 1 (off-by-one errors)
#         - Doubles or halves the result
#         - Drops decimal places
#     This creates a mismatch between the calculation and its stated result.
#     """
#     NAME = "EquationBalanceErrorModification"
    
#     @staticmethod
#     def is_applicable(text: str) -> bool:
#         # Look for equations with two equals signs (intermediate calculation = final result)
#         return bool(re.search(r'=.*?=\s*\d+(?:\.\d+)?(?:\s*(?:miles|mph))?(?:\n|$)', text))
    
#     @staticmethod
#     def apply(text: str) -> str:
#         # Find equations with two equals signs (to modify final results)
#         double_equals_pattern = r'(.*?=.*?=\s*)(\d+(?:\.\d+)?)(\s*(?:miles|mph)?(?:\n|$))'
#         # Or single equals with final result
#         single_equals_pattern = r'(=\s*)(\d+(?:\.\d+)?)(\s*(?:miles|mph)?(?:\n|$))'
        
#         # Try double equals first (for intermediate calculations)
#         matches = list(re.finditer(double_equals_pattern, text, re.MULTILINE))
#         if not matches:
#             # Fall back to single equals
#             matches = list(re.finditer(single_equals_pattern, text, re.MULTILINE))
#             if not matches:
#                 return text
        
#         # Choose a random equation to modify
#         match = random.choice(matches)
#         prefix, num_str, suffix = match.groups()
#         num = float(num_str.replace(',', ''))  # Remove commas for conversion
        
#         # Strategies for modifying the result
#         strategies = [
#             lambda x: x + 1,    # Off by one
#             lambda x: x - 1,    # Off by one (negative)
#             lambda x: x * 2,    # Double
#             lambda x: x / 2,    # Half
#         ]
        
#         new_num = random.choice(strategies)(num)
#         while new_num == num:
#             new_num = random.choice(strategies)(num)
        
#         # Format the new number to match original style
#         if '.' in num_str:
#             # Keep same number of decimal places
#             decimal_places = len(num_str.split('.')[1])
#             formatted_num = f"{new_num:.{decimal_places}f}"
#         else:
#             formatted_num = str(int(new_num))
        
#         # Never include commas in the result
#         formatted_num = formatted_num.replace(',', '')
        
#         return text[:match.start(2)] + formatted_num + text[match.end(2):]
    
class NothingModification(PerturbationStrategy):
    """Applies minimal text changes that preserve mathematical meaning.
    
    This is a fallback strategy when no mathematical perturbations are possible.
    Since one of the other strategies should be triggered when there's _any number_ in the text, I'm hoping this is never needed.
    
    Applicability:
        - Always applicable (fallback strategy)
    
    Perturbation:
        Returns the original text
    """
    NAME = "DoNothingModification"
    
    @staticmethod
    def is_applicable(text: str) -> bool:
        return True  # Always applicable as fallback
    
    @staticmethod
    def apply(text: str) -> str:
        return text

"""
# TODO: Sam... The idea is to add this stuff to the stub generation process (both on and off policy).
Note that I'll probably need to change my completion and verification columns to lm_completion and lm_verification
And my perturbed_stub to lm_perturbed stub (I should be doing this already, but it looks like Im not in the on policy, at least?)
"""
def get_perturbed_stub_deterministic(stub: str) -> Tuple[str, Optional[str]]:
    """
    Returns (perturbed_stub, perturbation_type)
    If no perturbation was possible, perturbation_type will be None
    """
    strategies = [
        NumericModification,
        OperatorSwapModification,
        UnitModification,
        PercentageModification,
        FractionModification,
        OrderOfOperationsModification,
        # EquationBalanceErrorModification,
    ]
    
    # Find applicable strategies
    applicable = [s for s in strategies if s.is_applicable(stub)]
    if not applicable:
        applicable.append(NothingModification)
        
    # Choose and apply random strategy
    strategy = random.choice(applicable)
    perturbed = strategy.apply(stub)
    
    return perturbed, strategy.__name__