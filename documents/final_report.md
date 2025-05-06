# Final Project Report

## Professional Development in Computer Science - 3112

### Charlie Bailey (peba2926)

#### Introduction

"What I cannot create, I do not understand." — Richard Feynman.

The original goal of this project was fairly straightforward—to build a large language model entirely from scratch, closely following the tutorial laid out in  Sebastian Raschka's book [Build a Large Language Model From Scratch](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167) (BaLLM). The ambition behind this vision was to deeply understand the intricate design of transformer-based LLMs by constructing one myself. Ultimately, I aimed to create a functional GPT-like chatbot capable of generating coherent responses to user queries. However, midway through the project, I adjusted my focus toward gaining a deeper conceptual understanding of the architecture rather than simply regurgitating code without fully understanding how it works. Throughout this report, I'll describe the rationale of this pivot, explore the knowledge and insights gained, and reflect on the lessons learned in failing to achieve my original goal.

#### Background

At its core, this project is about learning by building. In my opinion, the transformer architecture has been one of the most pivotal technological advancements in the past 25 years. Transformer-based systems like ChatGPT, Claude, and Gemini are already reshaping the landscape of human work. Ethan Mollick, a prominent professor at Wharton studying AI and innovation recently emphasized this point clearly in a post on X:

"I don’t mean to be a broken record but AI development could stop at the o3/Gemini 2.5 level and we would have a decade of major changes across entire professions & industries (medicine, law, education, coding…) as we figure out how to actually use it. AI disruption is baked in."

We are still in the early stages of comprehending the impacts that this technology will have on the future of human work. However, it already appears inevitable that transformer-based AI will play an increasingly central role in our work lives in the coming decade and beyond. As a student of computer science and artificial intelligence, I believe it is crucial to develop a deep understanding of how these technologies work. Acquiring this expertise will significantly enhance my ability to effectively leverage AI, ultimately increasing my own usefulness in the professional market. This belief forms the underlying rationale for pursuing this project as a part of my professional development.

#### Materials and Methods

For this project, I relied primarily on two resources. The first, as mentioned earlier, was Sebastian Raschka's book *Build a Large Language Model From Scratch*. This book served as the bedrock of my process and helped me frame the milestones that I would be working toward each week. I also relied heavily on the textbook *Speech and Language Processing* by Daniel Jurafsky and James H. Martin, which served as a complementary resource and offered a valuable alternate perspective on some of the more technically challenging aspects of transformer-based system architecture.

In addition to these texts, I was also very fortunate to be concurrently enrolled in a Natural Language Processing course taught by Professor Curry Guinn. The lectures and insights gained from this course significantly enhanced my learning and overall understanding throughout this project.

Aside from these main sources of material, I utilized a handful of other online resources, which I have cited in the references below.

To effectively acquire the skills necessary to complete this project, I adopted an approach inspired by the methodology of deliberate practice described by Barbara Oakley in her book, *A Mind for Numbers*. In her book, Oakley had the following observation on how a master violinist learns a new piece of music:

"A master violinist, for example, doesn’t just play a musical piece from beginning to end, over and over again. Instead, **she focuses on the hardest parts of the piece—the parts where the fingers fumble and the mind becomes confused**. You should be like that in your own deliberate practice, focusing and becoming quicker at the hardest parts of the solution procedures you are trying to learn."

While the analogy does not map perfectly to this project, the underlying idea—pausing deliberately at points of confusion and difficulty—greatly informed my approach to this project.

In practice, I would begin each project session by systematically working through the examples in the BaLLM book. Although only a few of these exercises directly contributed to the final implementation, they all proved to be invaluable in terms of building a true understanding of each aspect of the architecture. As I worked through each example, there would inevitably be points where the intricacies of the architecture were confusing. At this point, rather than glossing over that portion with a half-baked understanding, I would pause and focus on further breaking down the confusing concept or line of code until I understood exactly what was happening. Sometimes resolving this confusion would be as simple as looking up an unfamiliar method, but often it involved carefully whiteboarding mathematical transformations or seeking further elaboration in the *Speech and Language Processing* textbook.

This meticulous approach played a pivotal role in deepening my understanding, but it was also the reason I was unable to attain my final goal of developing a fully functional chatbot within my initial project timeline. I will further explore and reflect on this tension in subsequent sections of this report, but overall, this was one of the big learning lessons I took from this project.

#### Results

Throughout this project, I have learned each discrete step that takes place from passing in pieces of text, all the way through the contextualized embedding vector produced by the self-attention mechanism. I have successfully created working demonstrations for each step in this transformation. Furthermore, I can confidently walk through and clearly explain each line of code, describing precisely how it works and the role it serves in the overall architecture.

Although I did not achieve my initial project goal of generating a fully functional chatbot from scratch, I do feel that I have attained my mid-project pivot goal of developing a deep understanding of the LLM architecture up through the self-attention mechanism. Equipped with this knowledge, I feel well-positioned to continue this learning in future development of this project.

Ultimately, the true measure of this project's value will be its practical impact. When interviewing for a role as an AI engineer, my ability to clearly articulate my work and learning on this project will serve as the ultimate test of my understanding. While I feel confident that the work I have done so far has prepared me as much as possible, it is ultimately this future moment that will determine the true success of this project.

#### Reflection

I've reflected quite a bit on the outcomes of this project and what I could have done differently. I've vacillated between thinking it was a mistake to take on this project at all, to feeling satisfied that I at least took the initiative to begin a project that I might otherwise never have started. In the end, given the relatively low stakes, I am glad I chose to pursue this project despite not fully achieving my original goals. I am now more excited than ever at all of the possibilities ahead for further developing this project.

Looking back, I recognized from the beginning that time constraints were going to pose the biggest risk to completing the project. I ended up taking on the heaviest course load I've had in this program, including Intro to AI, which is quite possibly one of the most difficult courses I have ever taken. In hindsight, had I known how much work that single course would add to my plate, I honestly probably wouldn't have taken this professional development course at all. However, hindsight is always 20/20—the real challenge lies in accurately forecasting these challenges and mitigating them before they arise.

If I had been more diligent in planning my workload, it would have been beneficial to take a look at the number of hours per week other students reported for the courses I was taking and then consider that aggregate number in the context of my own working style and other obligations I had over the semester. Had I done this level of analysis and realistically evaluated the outcome, I likely would have concluded that taking on a project as complex as this was impractical.

On the other hand, despite these challenges, I'm ultimately pleased with the decision I made to pursue this project. As previously mentioned, had I not taken these initial steps, this project would have likely remained on my "want-to-do" list indefinitely. Now though, I've gotten the ball rolling and I'm more excited than ever to dedicate some serious time to completing this project. From this perspective, I would consider this project a meaningful success.

#### Conclusion

In conclusion, while I wasn't able to give this project the full attention it deserved, I am pleased with the progress and level of understanding I've gained given the constraints. I now possess a significantly deeper understanding of how LLMs work, and more importantly, I have a ton of motivation and excitement about the future development of this project.

Moving forward, my plan is to continue following the process outlined in BaLLM, albeit with a lot more time to dedicate to it. My goal remains to complete a working GPT-2 level personal assistant. One of the aspects I'm particularly excited about is the prospect of pre-training this model entirely from scratch. After some preliminary research into GPU compute costs, I've determined that it should be feasible to train a small GPT-2 level model on my own. While the BaLLM tutorial has the option to import pre-trained weights and focus solely on fine-tuning, I believe mastering dataset curation and managing an extended pre-training process (even at a small scale) are highly valuable skills to develop. Overall, I feel empowered and eager to carry this project forward, and I am excited about all of the possibilities that lie ahead.

#### References

3Blue1Brown. (n.d.). Neural networks. Retrieved May 5, 2025, from [https://www.3blue1brown.com/topics/neural-networks](https://www.3blue1brown.com/topics/neural-networks)

Karpathy, A. (2022, December 27). *Let’s build GPT: From scratch, in code, spelled out* [Video]. YouTube. [https://www.youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)

Jurafsky, D., & Martin, J. H. (2023). *Speech and language processing: An introduction to natural language processing, computational linguistics, and speech recognition with language models (3rd ed., draft)*. Stanford University & University of Colorado at Boulder.

Raschka, S. (2023). *Build a large language model (from scratch)* [Kindle ed.]. Amazon Kindle Direct Publishing.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention is all you need*. In Advances in Neural Information Processing Systems 31 (NIPS 2017). Retrieved from [https://papers.nips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf](https://papers.nips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
