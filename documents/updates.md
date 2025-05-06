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


### Week 5:
#### What did you do last week?

I spent this past week getting up to speed on transformers and the GPT architecture. I read the first chapter of BaLLM, the seminal paper on transformers, "Attention is all you need," and got about a quarter of the way through Andrej Karpathy's "Let's Build GPT" tutorial.

I also spent a fair amount of time building out my portfolio. Since this project is intended to be a portfolio piece, it makes sense to spend some time building out the rest of my portfolio to present it in the best possible light.

#### What do you plan to do this week?

This coming week, I plan to work through chapter 2 of BaLLM, which focuses on understanding how to work with text data. I also want to set up the environment a bit more and continue working through the Karpathy tutorial. A stretch goal for this coming week is to look into the coding work for the attention mechanism and potentially start setting up some of the scaffolding.

#### Are there any impediments in your way?

None at the moment.

#### Reflection on the process you used last week, how can you make the process work better?

For the last three weeks, I've basically been doing all of my work during a single 4-hour period on Thursday night. While this has been working in terms of staying on track, I'm finding the time block to be a bit constrained—there's just more that I want to work on. This week, I'm going to try breaking up the work into two 2-3 hour blocks to see if that allows me to be more productive and revisit specific points that I want to follow up on.


### Week 6-7:
#### What did you do last week?

It's been a rough two weeks. A bunch of things ended up coalescing on my schedule at once, so I ended up not being able to make any progress in week 6, which was a bummer. But I was able to carve out some time this past week and am finally into the build phase, which feels good. At this stage, I'm basically just following along with the BaLLM book in a more tutorial-like style. I set up a preprocessing pipeline that takes in a single text, converts it to individual tokens using regex, creates a unique vocabulary, and then assigns each token a unique token ID.

#### What do you plan to do this week?

Next week, I plan to finish the text preprocessing phase by creating token embeddings and encoding word position. If that goes quickly, I will move into coding the attention mechanism.

As luck would have it, we are just now getting into the transformer portion on the NLP course and I am extremely grateful to have that content to lean on for this project. I tend to get overwhelmed by complexity, so having Professor Guinn's lectures to reference in this project have been immensely helpful.

#### Are there any impediments in your way?

Right now, my biggest conceptual hurdle is trying to understand which parts of the BaLLM book simply introduce a concept and which parts I actually need to build and integrate into this project. As I was working through this chapter this past week, there was a point where the author said, "We will be using a more robust preprocessor in our implementation," nothing else was introduced, and the next chapter goes right into coding the attention mechanism. Parsing out exactly what needs to be built is definitely the most difficult challenge I am facing right now.

#### Reflection on the process you used last week, how can you make the process work better?

At this point, I'm just trying to keep my head above water. It looks like things should lighten up a bit next week, so hopefully, I won't be too pressed to get my full 4 hours in.


### Week 8:
#### What did you do last week?

This week I continued to work on the text preprocessing pipeline. I realize that a lot of the work I did last week ended up being unnecessary—which was disappointing. In building along with the BaLLM book, I'm realizing now that the author alternates between educational exercises to better understand the underlying mechanics and implementation of the actual code needed to make the LLM system work. I had been diligently following along with all the examples, but most of that work was unnecessary for the end product—which was disappointing.

Following this, I ended up re-building the tokenization preprocessing pipeline using a BPE tokenizer that I imported from an external library hosted by OpenAI. The library is called `tiktoken` and I utilized the BPE tokenizer used in GPT-2. I then created a simple data sampling sliding window that illustrated the context to next token prediction scheme Anon will be using (oh yeah, I also named my LLM Anon). After implementing this, I realized it was another educational exercise and not a functional final implementation. This sliding window exercise was just a preview for a more complex data loader implementation that utilizes the `Dataset` and `DataLoader` built-in classes from PyTorch. I ended this week learning more about these classes and tensors in general (which I will have to use for the data loader).

#### What do you plan to do this week?

Next week, I plan to finish the data loader—which is the final step in the preprocessing pipeline—and then begin working on building the attention mechanism. I realize I'm falling behind schedule, so I want to try and put in some additional hours this week to get back on track.

#### Are there any impediments in your way?

I need to decide how much time I want to devote to working through the educational exercise in the BaLLM book. I find the exercises very helpful for comprehension and understanding in terms of what I'm building, but they also take up a lot of time and don't contribute to my overall progress on the project. It's tough to know if I should "go slow to go fast" (i.e., take the time to deeply understand what I'm building to make the development process easier down the line) or charger ahead with the main components with a potentially less deep understanding of what I'm actually building. My normal style would be to go for the depth of understanding; however, I worry this will likely result in running out of time to fully complete the project.

#### Reflection on the process you used last week, how can you make the process work better?

I really want to try to break up my time block for this project into two periods next week. I keep saying I am going to do this, but then other obligations get in the way and I end up doing all of my work during a single time block. It seems like I have a little slack in my schedule this week though so I think I should be able to switch things up and see how it goes.


### Week 9:
#### What did you do last week?

I finally finished the preprocessing pipeline! The pipeline has a custom PyTorch Dataset class that processes raw text into overlapping "chunks" using a sliding window approach. It then tokenizes the text using the GPT-2 tokenizer, embeds the token in an embedding vector and then concatenates this with a positional vector—resulting in the final input embeddings that are ready to be passed into the transformer attention mechanism. Here are some screenshots of the interim output:

![Dataloader inputs and targets](./assets/dataloader_inputs_targets.png)

![Final input embeddings](./assets/input_embeddings.png)

#### What do you plan to do this week?

Next week I will get started on building the attention mechanism. I'm pretty excited for this part.

#### Are there any impediments in your way?

Nothing right now. Feels good to have achieved a tangible milestone.

#### Reflection on the process you used last week, how can you make the process work better?

Still stuck in the 4 hour block on Thursdays. I was actually able to get closer to 5 this week with the project proposal write up. I'm starting to think the 4 hours of deep work might actually be more desirable than attempting to break up the work period. There's something to be said for getting loaded into a project and going into flow state.


### Week 10:
#### What did you do last week?

This week I entered the belly of the beast—coding the transformer attention mechanism. In doing a first skim through the chapter, I will say I'm a little intimidated about the portion of the project. Luckily though, the BaLLM tutorial takes things one step at a time which definitely helps calm my "complexity anxiety."

This week ended up being very heavy on conceptual understanding. I spent a lot of my time whiteboarding the concepts that were being explained/coded. In the end, I was able to implement and conceptually wrap my head around a context vector and how it is computed. The three images below essentially mirror the process I went through this week–study the concept in the book, implement the concept in the code, whiteboard the concept to fully understand what each line of code is doing.

![Context Vector Calculation Diagram](./assets/context_vec_calc_diagram.png)
Raschka, Sebastian. Build a Large Language Model (From Scratch) (p. 124). (Function). Kindle Edition.

![Context Vector Code](./assets/context_vec_code_annotated.png)

![Context Vector Whiteboard](./assets/context_vec_whiteboard.png)

There is a lot going on in what amounts to not very many lines of code, so I feel like it's crucial to understand exactly what each line is doing so I don't get lost later on down the road.

#### What do you plan to do this week?

Next week and over the break I'm going continue building the self-attention mechanism. I'm hoping that once I fully understand the fundamentals of the attention mechanism, the rest of this portion of the project will go a bit more quickly. With some extra time over the break, I'm setting a stretch goal to have the attention mechanism completed by April 3rd.

#### Are there any impediments in your way?

Nothing specifically right now. Just trying not to get too overwhelmed by the complexity and continuing to focus on one step at a time.

#### Reflection on the process you used last week, how can you make the process work better?

I decided to really slow down and focus on conceptual understanding this week. This is a complex portion of the project, so I think it is worth spending a bit more time now rather than charging ahead without a solid grasp on what exactly it is that I'm building.



### Week 11:
#### What did you do last week?

This week I continued to work on coding the self-attention mechanism. I finished with the simple self-attention mechanism tutorial and have moved into the trainable self-attention mechanism. This mechanism introduces the concepts of query, key and value vectors—which took / are taking a bit of work to wrap my head around exactly how they operate. Following Professor Guinn's advice from my mid-semester report, I've started documenting my learning process as I go through these tutorials. I've also been using the `Speech and Language Processing. Daniel Jurafsky & James H. Martin.` textbook from the NLP class to help understand some of the conceptual hurdles like the query, key and value vectors. The goal here is to provide a clear, elaborate record for the learnings gained in each aspect of the project to formulate good talking points in an interview setting. 

#### What do you plan to do this week?

I'm planning to continue coding the attention mechanism. Flipping through the remainder of the chapter, I'm getting a little worried about how long this section of the project will take. I'm continuing to focus on deeply understanding each aspect I code, but with something as complex as a transformer, it just takes a lot of time to process.

#### Are there any impediments in your way?

The relentless march of time.

#### Reflection on the process you used last week, how can you make the process work better?

I don't have any notes in terms of improvements this week. I feel that I'm in a good place with my conceptual understandings. I just need to keep at it.



### Week 12-13:
#### What did you do last week?
Unfortunately I had another convergence of work/personal/school stuff hit my plate and wasn't able to work on the project last week. This week though, I was able to make up a bit of time and I'm *almost* done coding the LLM attention mechanism. I've implemented the 'simplified self-attention', 'trainable self-attention', and 'causal self-attention' mechanisms. The causal (or masked) self-attention mechanism is really the only one I will need to finish the project, but it was well worth the learning to go through the exercises of iteratively building up each portion of the attention mechanism one step at a time. In looking at the final compact code for the causal self-attention mechanism, I'm feeling very good about the decision to work through all of the learning example. The final class is only ~25 lines of code, but contained in that are so many matrix multiplications and data transformations that it would be almost impossible to understand what is going without having first broken each part out and implemented it individually line by line.

In addition to this, I've continued to update my Learning Process journal—taking the time to really understand and document the parts that were particularly confusing.

#### What do you plan to do this week?
This week is plan to finish the attention mechanism by extending the causal self-attention to be multi-head attention. Luckily, this part looks fair straight forward. After that, I'll begin working on the final full GPT architecture.

#### Are there any impediments in your way?
This is going to be a very difficult push to the end of the semester. In addition to two other AI/ML final projects, my best friend also decided that next weekend would be a good time to get married in the Bahamas. I literally don't think I could have picked a worse week in the semester if I tried. As it is, this class unfortunately tends to fall to the bottom of my priority queue. I'm going to try and put in as much time as I can, but these next two weeks are definitely going to be a grind to get everything completed on time.

#### Reflection on the process you used last week, how can you make the process work better?
No change in the previous process. The best thing I can do is spend more time working on this project to get everything I need to done.



### Week 14:
#### What did you do last week?
Apologies for the late update. As expected, last week ended up being very chaotic in terms of hitting all of my deadlines due to personal travel for my friends wedding. Luckily, I had ~6 hours of time to work on the plane ride back yesterday afternoon and was able to make some descent progress! I finally completed the self attention mechanism with the full implementation of the multi-headed causal attention class and have now moved on to building the full GPT architecture.

#### What do you plan to do this week?
At this point I have resigned myself to the realization that I will not be completing this project by the end of the semester. In hindsight, it was probably a mistake to taken on such an ambitious project on top of an already overly ambitious course load plus work obligations. That being said, I do feel this project as given me a much better understanding for how LLMs work. I also feel that—given sufficiently more time and attention—I will have no problem completing this project as I move into the job search. Despite biting off more than I could chew, I'm still glad I made the decision to pursue this project as this likely would have remained an item on my "want to do list" that never ended up getting started. Now, though, I have momentum and all of the tools and knowledge I need to get this project across the finish line and add a (hopefully) unique talking point to my portfolio as I begin my job search for an AI/ML engineering role.

This week, I intend to make one final push to get as much done as possible, then reflect on my project learnings in the final report.

#### Are there any impediments in your way?
Luckily the final report for this project isn't due until May 3rd, so I should have a good amount of time at the end of the week to make one final push on this project.

#### Reflection on the process you used last week, how can you make the process work better?
I knew this past week was going to be hectic in light of personal travel, but I actually found it to be quite zen working on the plane.


### Week 15:
#### What did you do last week?
I spent this past week attempting to find a good stopping point in my project. I'm still continuing to build out the main GPT architecture, so it's a sure thing at this point that I won't finish the project as initially planned. I still feel I've made good progress though and I've started working on cleaning up my "learnings" notebook. I'm trying to figure out the best way to present what I've done in the final project report and I think I want to formalize this notebook a bit more to include more prominently in my report.

Even though I'm not going to meet my original project goal, I am still very excited about this project and plan to continue working on it on the side as I enter the job hunt. It's been very fun thinking of all the little experiments I can try with my own little GPT model!

#### What do you plan to do this week?
This week I plan to wrap everything up and finish my final report.

#### Are there any impediments in your way?
Nope, it'll all be over by tomorrow night!

#### Reflection on the process you used last week, how can you make the process work better?
It's been pretty much the same work process this entire semester—this project gets shunted to the bottom of my priority pile and I end up doing all the work in a single time block. I'll have a lot to reflect on in my final report, but this has been a humbling lesson in setting reasonable expectations and not biting off more than you can chew.