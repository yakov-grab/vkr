"""
Example Python module with various functions to demonstrate test generation.
This file contains different types of functions to showcase how the test generator
handles various scenarios including numerical operations, string manipulations,
and more complex data structures.
"""

from typing import List, Dict, Optional, Union
from datetime import datetime


def add_numbers(a: float, b: float) -> float:
    """
    Add two numbers together and return the result.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of the two numbers
    """
    return a + b


def factorial(n: int) -> int:
    """
    Calculate the factorial of a non-negative integer.
    
    Args:
        n: Non-negative integer
        
    Returns:
        Factorial of n (n!)
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0:
        return 1
    
    result = 1
    for i in range(1, n + 1):
        result *= i
    
    return result


def is_palindrome(text: str) -> bool:
    """
    Check if a string is a palindrome (reads the same backward as forward).
    Case-insensitive and ignores non-alphanumeric characters.
    
    Args:
        text: Input string to check
        
    Returns:
        True if the string is a palindrome, False otherwise
    """
    # Remove non-alphanumeric characters and convert to lowercase
    cleaned_text = ''.join(char.lower() for char in text if char.isalnum())
    return cleaned_text == cleaned_text[::-1]
