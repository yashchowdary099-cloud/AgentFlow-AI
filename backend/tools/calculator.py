"""
Safe Math Calculator Tool
Uses Python AST (Abstract Syntax Tree) to parse and evaluate mathematical expressions safely.
Arbitrary code execution and builtins are strictly prohibited.
"""

import ast
import operator
import math
import re
from typing import Union, Dict, Any

# Allowed operators mapping
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed math functions
FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pi": math.pi,
    "e": math.e,
}

def _eval_node(node):
    """Recursively evaluate an AST node safely."""
    if isinstance(node, ast.Constant):  # Python 3.8+ numbers/constants
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
        
    elif isinstance(node, ast.BinOp):  # e.g., a + b, a * b
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type in OPERATORS:
            # Safe division check
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ZeroDivisionError("Division by zero is not allowed.")
            # Guard against massive exponent operations
            if op_type == ast.Pow and (right > 1000 or left > 10000):
                raise ValueError("Exponent too large to evaluate safely.")
            return OPERATORS[op_type](left, right)
        raise ValueError(f"Operator {op_type.__name__} not supported.")
        
    elif isinstance(node, ast.UnaryOp):  # e.g., -a, +a
        operand = _eval_node(node.operand)
        op_type = type(node.op)
        if op_type in OPERATORS:
            return OPERATORS[op_type](operand)
        raise ValueError(f"Unary operator {op_type.__name__} not supported.")
        
    elif isinstance(node, ast.Call):  # e.g., sqrt(144)
        if isinstance(node.func, ast.Name):
            func_name = node.func.id.lower()
            if func_name in FUNCTIONS and callable(FUNCTIONS[func_name]):
                args = [_eval_node(arg) for arg in node.args]
                return FUNCTIONS[func_name](*args)
            raise ValueError(f"Function '{func_name}' is not allowed.")
        raise ValueError("Invalid function call structure.")
        
    elif isinstance(node, ast.Name):  # e.g., pi, e
        name = node.id.lower()
        if name in FUNCTIONS and not callable(FUNCTIONS[name]):
            return FUNCTIONS[name]
        raise ValueError(f"Variable '{node.id}' not allowed.")
        
    else:
        raise ValueError(f"Unsupported expression syntax: {type(node).__name__}")

def preprocess_expression(raw_expr: str) -> str:
    """
    Clean and convert natural math phrases into valid arithmetic expressions.
    Handles phrases like '18% of 1250' -> '(18 / 100) * 1250'
    """
    expr = raw_expr.strip()
    
    # Remove leading command words if any (e.g. "Calculate 458 * 92", "Solve: 10 + 5")
    expr = re.sub(r'^(calculate|compute|solve|what is|find|evaluate)\s*[:]?\s*', '', expr, flags=re.IGNORECASE)
    
    # Handle percentage phrase: "X% of Y" -> "((X/100) * Y)"
    expr = re.sub(r'(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)', r'((\1 / 100) * \2)', expr, flags=re.IGNORECASE)
    
    # Handle standalone percentage: "X%" -> "(X / 100)"
    expr = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1 / 100)', expr)
    
    # Replace '^' with '**' for powers
    expr = expr.replace('^', '**')
    
    # Replace unicode multiplication/division symbols
    expr = expr.replace('×', '*').replace('÷', '/')
    
    # Strip trailing punctuation such as question marks or periods
    expr = expr.rstrip('?. \t\n\r')
    
    return expr

def calculate(expression: str) -> Dict[str, Any]:
    """
    Safely evaluate a mathematical expression and return the formatted result.
    
    Returns:
        dict: {
            "success": bool,
            "expression": str,
            "result": str,
            "numeric_result": float | int | None,
            "error": str | None
        }
    """
    try:
        clean_expr = preprocess_expression(expression)
        if not clean_expr:
            return {
                "success": False,
                "expression": expression,
                "result": "Empty mathematical expression.",
                "numeric_result": None,
                "error": "Empty expression"
            }
            
        # Parse expression into an AST
        parsed = ast.parse(clean_expr, mode='eval')
        numeric_val = _eval_node(parsed.body)
        
        # Format output neatly
        if isinstance(numeric_val, float):
            # If float is practically an integer (e.g. 42.0), format as int
            if numeric_val.is_integer():
                formatted_str = f"{int(numeric_val):,}"
                numeric_val = int(numeric_val)
            else:
                formatted_str = f"{numeric_val:,.4f}".rstrip('0').rstrip('.')
        elif isinstance(numeric_val, int):
            formatted_str = f"{numeric_val:,}"
        else:
            formatted_str = str(numeric_val)
            
        return {
            "success": True,
            "expression": clean_expr,
            "result": formatted_str,
            "numeric_result": numeric_val,
            "error": None
        }
    except ZeroDivisionError:
        return {
            "success": False,
            "expression": expression,
            "result": "Error: Division by zero is undefined.",
            "numeric_result": None,
            "error": "Division by zero"
        }
    except Exception as exc:
        return {
            "success": False,
            "expression": expression,
            "result": f"Could not evaluate mathematical expression: {str(exc)}",
            "numeric_result": None,
            "error": str(exc)
        }
