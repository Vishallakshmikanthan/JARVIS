from __future__ import annotations

import ast
import math
import operator
from loguru import logger

# ---------------------------------------------------------------------------
# Calculator Skill (AST-based safe math)
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.BitXor: operator.xor,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCS = {
    "math": math,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "log": math.log,
}

def _safe_eval(node):
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.BinOp):
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        func = _SAFE_FUNCS[node.func.id]
        args = [_safe_eval(arg) for arg in node.args]
        return func(*args)
    raise ValueError(f"Unsupported math syntax node: {type(node)}")

def calculate(user_input: str) -> str:
    """
    Perform safe math calculations.
    Example: "What is 15 + 32 * [1/2]?"
    """
    expr = user_input.replace("[", "(").replace("]", ")").lower()
    # Basic cleaning to extract numeric/math expression
    for word in ("calculate", "compute", "what is", "evaluate"):
        expr = expr.replace(word, "").strip()

    try:
        # Standardize 'plus' / 'minus' / 'times' / 'divided by'
        expr = expr.replace("plus", "+").replace("minus", "-").replace("times", "*").replace("divided by", "/")
        
        tree = ast.parse(expr, mode='eval')
        result = _safe_eval(tree.body)
        
        return f"The result is {result}."

    except Exception as e:
        logger.error(f"Calculator error: {e}")
        return "I encountered a problem performing that calculation. Please ensure it's a valid math expression."
