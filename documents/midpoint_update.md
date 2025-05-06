# Project Deep Understanding
## Mid-point Update

#### Developer

Charlie Bailey

#### Reflection on Project Goals

The initial vision for this project was to create a large language model entirely from scratch. While this still remains the desired deliverable from this project, I have shifted this slightly to more clearly align with the process I have gone through over the past 5 weeks. While it is a subtle pivot, I think the intention is important. My vision for this project now is to **deeply and fundamentally understand how LLM work by building one from scratch.** While the deliverable remains the same, the goal state is slightly more nuanced.

As I have been following the tutorial laid out in Sebastian Raschka's book [Build a Large Language Model From Scratch](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167) (BaLLM), I've realized that most of the core code to generate a functioning LLM has already been written. I could probably go through the book in a weekend or a few days, cherry-pick the most important structural components and having a deliverable that meets the goal of having a working LLM that generates conversational text. But I would have learned nothing in the process! I've realized that the real value in this project will be the **understanding** that comes from working through this tutorial book bit by bit—pausing at the appropriate points to fully understand the code I'm writing rather than just regurgitating Sebastian's work.

In this first 5 weeks, I would estimate that 70% of the code I've written ended up only being for educational purposes and will not be used in the final implementation. At first, I was feeling very demoralized at the pace of my progress. I was considering skipping all of the educational examples and only focusing on the true structural components. But then I took a step back and realized that the end product wasn't the actual goal of this project. Simply putting the click bait headline of "Built an LLM from scratch" on my portfolio might get me a few more job interviews, but it would all be for not if I didn't have the technical chops to fully explain exactly what was going on in each portion of the final implementation.

So, the TLDR of the above is—the project remains on track, slightly behind schedule, but that's ok because I'm giving myself permission to take my time and deeply understand how LLMs work at a foundational level.

#### Milestones Update

We are currently in week 5 of this project. Here is where I stand on my original milestones timeline:

| Weeks | Goal | Status | 
|------|------|-------|
| 1 | Setup environment and get familiar with PyTorch | Done |
| 2 | Understand the architecture; read the "Attention is all you need" paper | Done | 
| 3 | Learn how to work with text data; tokenization, byte-pair encoding, etc | Done |
| 4-5 | Code the attention mechanism | In progress | 
| 6-8 | Implement the GPT architecture | Not done |
| 9 | Pretrain on unlabeled data | Not done |
| 10 | Fine-tune the model | Not done | 
| 11-12 | Final polishing and buffer time for overruns | Not done |

When I initially set out these milestones, I didn't realize how much work would need to be on on the preprocessing front. There ended up being a lot more coding involved in this stage than initially anticipated. Overall, I would say I'm probably a week or two behind this initial timeline. Luckily, I built in two weeks on the end for overruns and also allocated a significant amount of time for coding the attention mechanism and overall architecture. I won't know until I get further into it, but these sections may be easier to implement than I initially anticipated. One thing I've realized from the "go slow to go fast" method is that things can seem very slow at first, but then knowledge compounds and progress can occur exponentially in the later stages of a project. I'm optimistic that I can attain the final vision of having a locally hosted, completely hand-built, GPT-like chatbot that I can query and generates coherent responses.
