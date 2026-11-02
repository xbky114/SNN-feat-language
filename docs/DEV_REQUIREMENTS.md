# requirements
1. Use conda 'llava' environment
2. Pay attention to code readability. Do not use ';' to seperate lines. Just create a new line.
3. Focus on extendability. The design of the codebase provides API to be compatible with different CLIP models and different conversion methods. This philosophy shoubd be maintained through the whole development. Write a factory rather than picking up specific names.
4. Your response should be very precise and concise without redundant, stupid, irrelavant, not-self-contained words.
5. When a development plan is decided, strictly follow it. Do not implement less, and do not implement redundant features.
6. Scripts which are not related to the core function of this codebase should be placed at tests/
7. Unless required, do not change commited docs; Only create a new doc when content cannot be viewed as a chapter of existing doc.