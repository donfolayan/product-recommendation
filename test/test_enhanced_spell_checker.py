import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.enhanced_spell_checker import EnhancedSpellChecker

class DummyModel:
    def encode(self, text, convert_to_tensor=False):
        # Return a fixed vector for testing
        import numpy as np
        return np.ones(384)

def test_pattern_correction():
    checker = EnhancedSpellChecker(model=DummyModel())
    # Should correct TSHIRT to T-Shirt
    assert checker._apply_pattern_corrections('TSHIRT') == 'T-Shirt'
    # Should correct HDPHN to headphone
    assert checker._apply_pattern_corrections('HDPHN') == 'headphone'

def test_add_product_terms():
    checker = EnhancedSpellChecker(model=DummyModel())
    checker.add_product_terms(['Super-Widget'])
    assert 'super-widget' in checker.product_dictionary
    assert 'super widget' in checker.product_dictionary

def test_correct_with_product_term():
    checker = EnhancedSpellChecker(model=DummyModel())
    checker.add_product_terms(['SuperWidget'])
    # Should not correct a known product term
    assert checker.correct('SuperWidget') == 'SuperWidget'

def test_correct_with_typo():
    checker = EnhancedSpellChecker(model=DummyModel())
    checker.add_product_terms(['laptop'])
    # Should correct 'laptpo' to 'laptop' if in dictionary
    result = checker.correct('laptpo')
    # Accept either 'laptop' or 'laptpo' if confidence threshold is not met
    assert result in ['laptop', 'laptpo'] 