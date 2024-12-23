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
        - Examples: "5", "3.14", "100"
    
    Perturbation:
        Applies one of several transformations to a randomly chosen number:
        - Increase by 50% (multiply by 1.5)
        - Decrease by 50% (multiply by 0.5)
        - Increase by order of magnitude (multiply by 10)
        - Decrease by order of magnitude (multiply by 0.1)
        - Change sign (multiply by -1)
    """
    NAME = "NumericModification"

    @staticmethod
    def is_applicable(text: str) -> bool:
        return bool(re.search(r'\d+(?:\.\d+)?', text))
    
    @staticmethod
    def apply(text: str) -> str:
        numbers = re.finditer(r'\d+(?:\.\d+)?', text)
        numbers = list(numbers)
        if not numbers:
            return text
        
        target = random.choice(numbers)
        num = float(target.group())
        
        strategies = [
            lambda x: x * 1.5,  # Increase by 50%
            lambda x: x * 0.5,  # Decrease by 50%
            lambda x: x * 10,   # Order of magnitude up
            lambda x: x * 0.1,  # Order of magnitude down
            lambda x: -x,       # Change sign
        ]
        new_num = random.choice(strategies)(num)
        
        return text[:target.start()] + str(new_num) + text[target.end():]

class OperatorSwapModification(PerturbationStrategy):
    """
    Swaps mathematical operators with their opposites.
    
    Applicability:
        - Text contains any of these operator pairs:
        - Addition/subtraction (+/-)
        - Multiplication/division (*/)
        - Greater/less than (>/< or >=/<= )
    
    Perturbation:
        Replaces one operator with its opposite:
        - + → -
        - * → /
        - > → <
        - >= → <=
    """
    NAME = "OperatorSwapModification"

    OPERATOR_PAIRS = [
        (r'\+', '-'),
        (r'\*', '/'),
        (r'>', '<'),
        (r'>=', '<='),
    ]
    
    @staticmethod
    def is_applicable(text: str) -> bool:
        return any(re.search(op[0], text) for op in OperatorSwapModification.OPERATOR_PAIRS)
    
    @staticmethod
    def apply(text: str) -> str:
        applicable_pairs = [
            pair for pair in OperatorSwapModification.OPERATOR_PAIRS 
            if re.search(pair[0], text)
        ]
        if not applicable_pairs:
            return text
            
        op_pair = random.choice(applicable_pairs)
        return re.sub(op_pair[0], op_pair[1], text, count=1)

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
    Modifies mathematical expressions by changing parentheses placement.
    
    Applicability:
        - Text contains arithmetic expressions with parentheses
        - Example: "(2 + 3) * 4"
    
    Perturbation:
        Either:
        - Removes parentheses entirely
        - Shifts parentheses to change operation order
        This changes the order of operations and thus the result.
    """
    NAME = "OrderOfOperationsModification"

    @staticmethod
    def is_applicable(text: str) -> bool:
        return bool(re.search(r'\([^()]+\)', text))
    
    @staticmethod
    def apply(text: str) -> str:
        matches = list(re.finditer(r'\(([^()]+)\)', text))
        if not matches:
            return text
            
        match = random.choice(matches)
        strategies = [
            lambda expr: expr.group(1),  # Remove parentheses
            # Shift parentheses left if possible
            lambda expr: f"({expr.group(1).split()[0]}) {' '.join(expr.group(1).split()[1:])}"
        ]
        
        new_expr = random.choice(strategies)(match)
        return text[:match.start()] + new_expr + text[match.end():]
    

class EquationBalanceErrorModification(PerturbationStrategy):
    """
    Introduces errors in equation results while preserving the calculation.
    
    Applicability:
        - Text contains an equation where the right side is a number
        - Example: "2 + 2 = 4", "3 * 5 = 15"
    
    Perturbation:
        Modifies only the result (right side) while keeping the calculation:
        - Adds or subtracts 1 (off-by-one errors)
        - Doubles or halves the result
        - Drops decimal places
        This creates a mismatch between the calculation and its stated result.
    """
    NAME = "EquationBalanceErrorModification"
    
    @staticmethod
    def is_applicable(text: str) -> bool:
        # Look for patterns like "X = Y" where Y is a number
        return bool(re.search(r'=\s*\d+(?:\.\d+)?', text))
    
    @staticmethod
    def apply(text: str) -> str:
        # Find all equations where the right side is a number
        matches = list(re.finditer(r'(=\s*)(\d+(?:\.\d+)?)', text))
        if not matches:
            return text
        
        # Choose a random equation to modify
        match = random.choice(matches)
        num = float(match.group(2))
        
        # Strategies for modifying the result
        strategies = [
            lambda x: x + 1,    # Off by one
            lambda x: x - 1,    # Off by one (negative)
            lambda x: x * 2,    # Double
            lambda x: x / 2,    # Half
            lambda x: int(x),   # Drop decimal
        ]
        
        new_num = random.choice(strategies)(num)
        return text[:match.start(2)] + str(new_num) + text[match.end(2):]
    
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
        EquationBalanceErrorModification,
    ]
    
    # Find applicable strategies
    applicable = [s for s in strategies if s.is_applicable(stub)]
    if not applicable:
        applicable.append(NothingModification)
        
    # Choose and apply random strategy
    strategy = random.choice(applicable)
    perturbed = strategy.apply(stub)
    
    return perturbed, strategy.__name__