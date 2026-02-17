import dataclasses
from typing import Optional, List

@dataclasses.dataclass
class Clue:
    text: str
    length: int
    pattern: Optional[str] = None
    type: str = "standard" # standard or cryptic

class CrosswordSolver:
    def __init__(self):
        print("Initializing Crossword Solver...")

    def generate_prompt(self, clue: Clue, strategy: str = "cot") -> str:
        """
        Generates a prompt for the LLM based on the selected strategy.
        Strategies: 'cot' (Chain of Thought), 'analogical', 'react'.
        """
        if strategy == "cot":
            return self._prompt_cot(clue)
        elif strategy == "analogical":
            return self._prompt_analogical(clue)
        elif strategy == "react":
            return self._prompt_react(clue)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _prompt_cot(self, clue: Clue) -> str:
        return f"""
I will provide you with a crossword clue and the number of letters in the answer. To solve this:
1.  Analyze the definition part of the clue.
2.  Analyze any wordplay (anagrams, homophones, hidden words).
3.  Brainstorm potential candidates that fit the letter count.
4.  Cross-check the candidates against the definition and wordplay.
5.  State the final answer.

Clue: {clue.text}
Length: {clue.length}
Pattern: {clue.pattern or "N/A"}
"""

    def _prompt_analogical(self, clue: Clue) -> str:
        return f"""
Here are examples of how to solve crossword clues:

**Example 1:**
Clue: 'Feline companion (3)'
Reasoning: The definition is 'feline companion'. A common 3-letter word for a cat is CAT.
Answer: CAT

**Now, solve this new clue following a similar process:**
Clue: {clue.text}
Length: {clue.length}
Pattern: {clue.pattern or "N/A"}
Reasoning:
"""

    def _prompt_react(self, clue: Clue) -> str:
         return f"""
You are an agent solving a crossword. You can use tools like `DictionarySearch` or `WordPatternSearch`.

**Question:** Solve the clue provided.

Clue: {clue.text}
Length: {clue.length}
Pattern: {clue.pattern or "N/A"}

**Thought 1:** I need to identify the definition and possible wordplay.
**Act 1:** Search for synonyms of the definition keyword.
"""

    def call_llm(self, prompt: str) -> str:
        """
        Mock LLM call. In a real implementation, this would call an API like Gemini or OpenAI.
        """
        print(f"\n--- Sending Prompt to LLM ---\n{prompt}\n-----------------------------")
        return "MOCK_ANSWER"

    def solve(self, clue: Clue, strategy: str = "cot"):
        print(f"Solving clue: '{clue.text}' ({clue.length}) using {strategy}...")
        prompt = self.generate_prompt(clue, strategy)
        response = self.call_llm(prompt)
        print(f"LLM Response: {response}")

if __name__ == "__main__":
    solver = CrosswordSolver()

    # Test Case 1: Standard Clue with CoT
    clue1 = Clue(text="Feline companion", length=3)
    solver.solve(clue1, strategy="cot")

    # Test Case 2: Cryptic Clue with Analogical
    clue2 = Clue(text="Mixed pear inside", length=4, type="cryptic")
    solver.solve(clue2, strategy="analogical")
