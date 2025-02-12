# Project Updates

### Week 3:
#### What did you do last week?
I spent this past week doing a search for some additional resources and then put together the project proposal. I also setup a project Trello board with some initial tasks and built a simple time-tracking spreadsheet.

#### What do you plan to do this week?
This week, I plan to start implementing on the timeline laid out in the proposal. I plan to:
* Read the BaLLM appendix on PyTorch
* Setup a suitable Docker Dev Container with all of the tools I'll need
* Get a "hello world" type program running in the new environment to make sure everything is working

#### Are there any impediments in your way?
I've found that environment setup can end up taking far longer than expected sometimes. I've only started using Docker in the past few months, so there may be some technical obstacles that I will have to overcome. PyTorch is a very well-supported library, though, so I'm sure there will be ample documentation if I need to troubleshoot.

#### Reflection on the process you used last week, how can you make the process work better?
This past week, I basically saved all of my work—with the exception of some early week planning—for Thursday night. While it's nice to have a dedicated time block to work on this project, I think going forward, I would like to break the work up a bit more throughout the week if possible.



### Week 4:
#### What did you do last week?
This week was all about getting my dev environment setup and getting familiar with PyTorch. As expected, this setup processes ended up being more time consuming than expected. I initially planned to use a Docker Dev Container in VS Code to get more practice with Docker, but after successfully creating a container, I realized I couldn’t use my MacBook Pro M3 GPU from inside Docker. Since GPU support is essential for training my LLM, I decided to scrap Docker and develop locally instead.

Switching to a local environment turned out to be straightforward:
* Used `pyenv` to download Python 3.10 (as recommended for PyTorch)
* Set this as the local version in my project directory
* Created a .venv and installed `pytorch` and `numpy` (keeping dependencies minimal)
* Added a main.py file
* Tested the environment using a supplementary script from BaLLM to confirm everything is working

#### What do you plan to do this week?
The plan for this coming week is to understand the architecture of LLMs. I plan to:
* Read the first chapter of BaLLM
* Read the paper "Attention is all you need."
* Start watching Andrej Karpathy's "Let's Build GPT" tutorial

The main goal this week is to get a high-level conceptual understanding for how LLMs work so I can better understand what exactly I will be building.

#### Are there any impediments in your way?
None at the moment.

#### Reflection on the process you used last week, how can you make the process work better?
Unfortunately, I was quite sick for a couple of days, which impacted my productivity. Considering the circumstances, I’m pleased with what I managed to accomplish. Hopefully, this week will be more productive.