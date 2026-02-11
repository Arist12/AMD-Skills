---
name: clean-code-style
description: >
  Behavioral guidelines to reduce common LLM coding mistakes. Use when writing, reviewing,
  or refactoring code to avoid overcomplication, make surgical changes, surface assumptions,
  and define verifiable success criteria. Derived from Andrej Karpathy's observations on
  LLM coding pitfalls. Applies universally regardless of language or domain.
---

# Clean Code Style

Behavioral guidelines to reduce common LLM coding mistakes, derived from Andrej Karpathy's observations on LLM coding pitfalls.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

---

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### Example: Hidden Assumptions

User: "Add a feature to export user data"

Bad — silently assumes scope, format, fields, and file location:
```python
def export_users(format='json'):
    users = User.query.all()
    with open('users.json', 'w') as f:
        json.dump([u.to_dict() for u in users], f)
```

Good — surfaces assumptions before writing code:
```
Before implementing, I need to clarify:
1. Scope: Export all users or filtered subset? (privacy implications)
2. Format: Download file? Background job? API endpoint returning data?
3. Fields: Which user fields? (some might be sensitive)
4. Volume: How many users typically? (affects approach)
```

---

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### Example: Over-abstraction

User: "Add a function to calculate discount"

Bad — strategy pattern, config dataclass, 50+ lines for one calculation:
```python
class DiscountStrategy(ABC):
    @abstractmethod
    def calculate(self, amount: float) -> float: ...

class PercentageDiscount(DiscountStrategy): ...
class FixedDiscount(DiscountStrategy): ...

@dataclass
class DiscountConfig:
    strategy: DiscountStrategy
    min_purchase: float = 0.0
    max_discount: float = float('inf')
```

Good — one function, done:
```python
def calculate_discount(amount: float, percent: float) -> float:
    return amount * (percent / 100)
```

Add complexity later when you actually need multiple discount types.

---

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: **every changed line should trace directly to the user's request.**

### Example: Drive-by Refactoring

User: "Fix the bug where empty emails crash the validator"

Bad — adds docstring, improves email validation, adds username validation nobody asked for:
```diff
  def validate_user(user_data):
-     # Check email format
-     if not user_data.get('email'):
+     """Validate user data."""
+     email = user_data.get('email', '').strip()
+     if not email:
          raise ValueError("Email required")
-     if '@' not in user_data['email']:
+     if '@' not in email or '.' not in email.split('@')[1]:
          raise ValueError("Invalid email")
+     username = user_data.get('username', '').strip()
+     if len(username) < 3:
+         raise ValueError("Username too short")
```

Good — only fixes the reported bug:
```diff
  def validate_user(user_data):
      # Check email format
-     if not user_data.get('email'):
+     email = user_data.get('email', '')
+     if not email or not email.strip():
          raise ValueError("Email required")
-     if '@' not in user_data['email']:
+     if '@' not in email:
          raise ValueError("Invalid email")
```

### Example: Style Drift

User: "Add logging to the upload function"

Bad — changes quote style, adds type hints, adds docstring, reformats whitespace:
```diff
- def upload_file(file_path, destination):
+ def upload_file(file_path: str, destination: str) -> bool:
+     """Upload file to destination with logging."""
```

Good — adds logging, matches existing single-quote style, touches nothing else:
```diff
+ import logging
+ logger = logging.getLogger(__name__)
+
  def upload_file(file_path, destination):
+     logger.info(f'Starting upload: {file_path}')
      try:
```

---

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

### Example: Test-First Verification

User: "The sorting breaks when there are duplicate scores"

Bad — immediately changes sort logic without confirming the bug exists.

Good — reproduce first, then fix:
```python
# 1. Write a test that reproduces the issue
def test_sort_with_duplicate_scores():
    scores = [
        {'name': 'Alice', 'score': 100},
        {'name': 'Bob', 'score': 100},
        {'name': 'Charlie', 'score': 90},
    ]
    result = sort_scores(scores)
    assert result[0]['score'] == 100
    assert result[1]['score'] == 100
    assert result[2]['score'] == 90

# Verify: test fails with inconsistent ordering

# 2. Fix with stable sort
def sort_scores(scores):
    return sorted(scores, key=lambda x: (-x['score'], x['name']))

# Verify: test passes consistently
```

---

## Anti-Patterns Summary

| Principle | Anti-Pattern | Fix |
|-----------|-------------|-----|
| Think Before Coding | Silently assumes format, fields, scope | List assumptions, ask for clarification |
| Simplicity First | Strategy pattern for one calculation | One function until complexity is needed |
| Surgical Changes | Reformats quotes, adds type hints during bug fix | Only change lines that fix the reported issue |
| Goal-Driven | "I'll review and improve the code" | "Write test for bug X -> make it pass -> verify no regressions" |

---

## Key Insight

The overcomplicated examples aren't obviously wrong — they follow design patterns and best practices. The problem is **timing**: they add complexity before it's needed.

Good code solves today's problem simply, not tomorrow's problem prematurely.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
