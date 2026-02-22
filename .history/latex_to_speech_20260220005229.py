import sympy
from sympy.parsing.latex import parse_latex
import re
from typing import Dict, List
import logging

class EquationToSpeech:
    """Converts LaTeX equations to natural language speech"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common mathematical expressions and their speech forms
        self.speech_patterns = {
            '^': ' to the power of ',
            '_': ' sub ',
            '\\frac': ' fraction ',
            '\\sqrt': ' square root of ',
            '\\sum': ' sum over ',
            '\\int': ' integral of ',
            '\\partial': ' partial derivative of ',
            '\\infty': ' infinity ',
            '\\alpha': ' alpha ',
            '\\beta': ' beta ',
            '\\gamma': ' gamma ',
            '\\theta': ' theta ',
            '\\lambda': ' lambda ',
            '\\mu': ' mu ',
            '\\sigma': ' sigma '
        }
    
    def equation_to_speech(self, latex_eq: str) -> str:
        """
        Convert LaTeX equation to natural language
        
        Args:
            latex_eq: LaTeX equation string
            
        Returns:
            Natural language description
        """
        try:
            # Clean and normalize LaTeX
            cleaned_eq = self._clean_latex(latex_eq)
            
            # Try symbolic parsing for complex equations
            try:
                expr = parse_latex(cleaned_eq)
                description = self._parse_sympy_expression(expr)
            except:
                # Fallback to pattern-based conversion
                description = self._pattern_based_conversion(cleaned_eq)
            
            return description
            
        except Exception as e:
            self.logger.error(f"Error converting equation: {e}")
            return f"Mathematical expression: {latex_eq}"
    
    def _clean_latex(self, latex: str) -> str:
        """Clean and normalize LaTeX string"""
        # Remove extra spaces
        latex = re.sub(r'\s+', ' ', latex.strip())
        
        # Handle common LaTeX patterns
        latex = re.sub(r'\\left|\\right', '', latex)
        
        return latex
    
    def _parse_sympy_expression(self, expr) -> str:
        """Parse sympy expression to natural language"""
        # Convert sympy expression to string and then to speech
        expr_str = str(expr)
        
        # Replace operators with words
        replacements = {
            '**': ' to the power of ',
            '*': ' times ',
            '/': ' divided by ',
            '+': ' plus ',
            '-': ' minus ',
            '=': ' equals ',
            'sqrt': 'square root of '
        }
        
        for op, replacement in replacements.items():
            expr_str = expr_str.replace(op, replacement)
        
        return expr_str
    
    def _pattern_based_conversion(self, latex: str) -> str:
        """Convert LaTeX to speech using pattern matching"""
        speech = latex
        
        # Replace LaTeX commands with speech equivalents
        for cmd, speech_form in self.speech_patterns.items():
            speech = speech.replace(cmd, speech_form)
        
        # Handle fractions specially
        speech = self._handle_fractions(speech)
        
        # Handle subscripts and superscripts
        speech = self._handle_superscripts_subscripts(speech)
        
        # Clean up multiple spaces
        speech = re.sub(r'\s+', ' ', speech)
        
        return speech.strip()
    
    def _handle_fractions(self, text: str) -> str:
        """Handle fraction patterns"""
        # Find \frac{numerator}{denominator} patterns
        fraction_pattern = r'\\frac\{([^}]+)\}\{([^}]+)\}'
        
        def fraction_replacer(match):
            num, den = match.groups()
            return f" {num} divided by {den} "
        
        return re.sub(fraction_pattern, fraction_replacer, text)
    
    def _handle_superscripts_subscripts(self, text: str) -> str:
        """Handle superscripts and subscripts"""
        # Handle ^ for superscripts
        text = re.sub(r'\^\{([^}]+)\}', r' to the power of \1 ', text)
        text = re.sub(r'\^(\w)', r' to the power of \1 ', text)
        
        # Handle _ for subscripts
        text = re.sub(r'\_\{([^}]+)\}', r' sub \1 ', text)
        text = re.sub(r'\_(\w)', r' sub \1 ', text)
        
        return text