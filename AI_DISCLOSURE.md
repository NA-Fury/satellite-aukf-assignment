### 7. **AI_DISCLOSURE.md**

```markdown
# AI Tool Disclosure

In accordance with the assignment requirements, this document discloses the use of AI-based tools in completing this project.

## Tools Used

1. **GitHub Copilot**: Used for code completion and boilerplate generation
   - Assisted with: Basic function signatures and docstring templates
   - Percentage of contribution: ~5%

2. **ChatGPT**: Used for debugging and documentation
   - Assisted with:
     - Debugging numpy broadcasting issues in sigma point calculation
     - Formatting matplotlib plots
     - README structure suggestions
   - Percentage of contribution: ~10%

## Original Work Declaration

The following components are entirely my own work:
- Algorithm design and selection (Sage-Husa adaptive approach)
- Filter architecture and implementation strategy
- Parameter tuning methodology
- Integration with Orekit propagator
- Performance evaluation metrics
- All mathematical derivations and design decisions

## Specific Contributions

### AI-Assisted Components:
1. **Sigma point generation** (lines 89-112 in aukf.py): ChatGPT helped debug index error
2. **Plot formatting** (visualization.py): Copilot suggested matplotlib parameters
3. **Test fixtures** (test_aukf.py): Copilot generated pytest fixture templates

### Fully Original Components:
1. Complete adaptive noise estimation algorithm
2. Outlier rejection logic
3. Orekit integration wrapper
4. All notebooks and analysis
5. Design documentation and rationale

I can explain and defend all aspects of the implementation, including the AI-assisted portions.
