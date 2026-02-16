# Crossword Solver Chatbot: Prompt Strategies

This document outlines three key prompting strategies for a crossword puzzle assistant: Chain of Thought (CoT), Analogical Prompting, and ReAct.

## 1. Chain of Thought (CoT)
**Objective:** Encourage the model to break down valid clues step-by-step before answering.

**Prompt Structure:**
> "I will provide you with a crossword clue and the number of letters in the answer. To solve this:
> 1.  Analyze the definition part of the clue.
> 2.  Analyze any wordplay (anagrams, homophones, hidden words).
> 3.  Brainstorm potential candidates that fit the letter count.
> 4.  Cross-check the candidates against the definition and wordplay.
> 5.  State the final answer.
>
> Clue: [Insert Clue]
> Length: [Insert Length]"

## 2. Analogical Prompting
**Objective:** Use successful examples of solved clues to guide the model's reasoning for new clues.

**Prompt Structure:**
> "Here are examples of how to solve crossword clues:
>
> **Example 1:**
> Clue: 'Feline companion (3)'
> Reasoning: The definition is 'feline companion'. A common 3-letter word for a cat is CAT.
> Answer: CAT
>
> **Example 2:**
> Clue: 'Mixed pear inside (4)'
> Reasoning: 'Mixed' suggests an anagram. Anagram of 'pear' is 'reap' or 'pare'. 'Inside' might simply mean the word is hidden or related. 'Mixed pear' strongly suggests REAP. Let's check: R-E-A-P (4 letters).
> Answer: REAP
>
> **Now, solve this new clue following a similar process:**
> Clue: [Insert Clue]
> Length: [Insert Length]
> Reasoning:"

## 3. ReAct (Reason + Act)
**Objective:** Allow the agent to reason about a clue, then 'act' (e.g., search a dictionary, check letter constraints) before finalizing an answer.

**Prompt Structure:**
> "You are an agent solving a crossword. You can use tools like `DictionarySearch` or `WordPatternSearch`.
>
> **Question:** Solve the clue provided.
>
> **Thought 1:** I need to identify the definition and possible wordplay.
> **Act 1:** Search for synonyms of the definition keyword.
> **Obs 1:** [Tool Output]
>
> **Thought 2:** I have a list of synonyms. Now I verify which one fits the length constraint and intersecting letters.
> **Act 2:** Check synonyms against the length [Length].
> **Obs 2:** [Tool Output]
>
> **Thought 3:** I have found the best match.
> **Final Answer:** [Answer]"

---
*Choose the strategy that best fits the complexity of the clue or the available tools.*
