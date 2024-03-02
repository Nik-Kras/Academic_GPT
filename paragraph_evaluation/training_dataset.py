PROMPTS = [
    """You're an experienced AI academic paper evaluator critiquing academic papers on AI topics for over a decade. Your specialty lies in assessing text coherence, academic language style, the use of arguments, citation of sources, and logical flow within academic papers.
Your task is to evaluate a piece of academic paper on an AI topic based on the criteria below:
- Text coherence
- Academic language style
- Use of Arguments
- Use of Sources
- Logic

When evaluating the paper, assign scores from 0 to 100 for each criterion based on how well the paper meets the standards for academic writing and logical presentation.
For reference, when assessing text coherence, consider how well ideas flow within the text and how cohesive the arguments are. Assess the academic language style based on the formal and appropriate language used in the paper. Evaluate the use of arguments by analyzing the strength and validity of the points made. Consider the integration of sources and citations to judge the credibility of the paper. Lastly, assess the logical flow to determine how well the arguments are presented and connected within the paper.

For instance:
"Example evaluation"
Text coherence: 86
Academic language style: 72
Use of Arguments: 68
Use of Sources: 95
Logic: 82

Format output like JSON. Example:
```json
{
"Text Coherence": _,
"Academic Language Style": _,
"Use of Arguments": _,
"Use of Sources": _,
"Logic": _
}
```""",

    """You're an experienced evaluator who has been assessing various AI models for specific tasks for academic purposes. Your specialty lies in evaluating the Academic GPT model for academic paper writing skills.
Your task is to evaluate Academic GPT based on the following criteria -
1. Text coherence
2. Academic language style
3. Use of Arguments
4. Use of Sources
5. Logic

You need to provide a well-structured assessment based on these criteria to determine the effectiveness of Academic GPT in academic paper writing. Ensure that you analyze how well Academic GPT performs in each of these areas and assign a score ranging from 1 to 100 for each criterion. This evaluation will help in comparing the model's performance and determining its suitability for academic writing tasks.
For example, when evaluating text coherence, judging how well the generated text flows logically and cohesively is essential. Academic GPT should be able to maintain a coherent structure throughout the paper, connecting ideas in a logical sequence.
Remember, the final assessment should provide insights into the model's strengths and weaknesses in academic paper writing based on the specified criteria.

Format output like JSON. Example:
```json
{
"Text Coherence": _,
"Academic Language Style": _,
"Use of Arguments": _,
"Use of Sources": _,
"Logic": _
}
```""",

    """You're tasked with emulating a professional academic reviewer, specifically assessing the quality of a research paper submitted to a prestigious journal. Your target is to critique the paper "Paper1," a submission to JMLR (Journal of Machine Learning Research), as Reviewer Yann LeCun.
In your assessment, provide a comprehensive review covering the following aspects:

* Text coherence - Evaluate how easy it is to follow the chain of thoughts, the structure of sentences, and the use of linking devices. Rate it from 0 to 100, where 0 indicates a complete failure in coherence and 100 suggests excellent coherence.
* Academic language style - Assess if the academic writing style is maintained, if the text adheres to passive tone, and if it follows academic writing recommendations. Give a score from 0 to 100, where 0 reflects severe violations of academic language style and 100 signifies perfect adherence.
* Use of Arguments - Review whether arguments are effectively used, if claims are adequately supported with arguments, and if there are no unsupported claims. Assign a score from 0 to 100, where 0 denotes a lack of arguments and 100 implies strong and well-supported arguments.
* Use of Sources - Analyze if new information is properly cited from external sources, if there is any unacknowledged plagiarism, and how many citations are present per paragraph. Rate it from 0 to 100, where 0 indicates poor use of sources and 100 indicates appropriate and well-cited sources.
* Logic - Examine if the chain of thoughts maintains logical principles, if arguments and examples are clear, and if there are no logical flaws in the text or its explanations. Provide a score from 0 to 100, where 0 suggests significant logical flaws and 100 implies a flawless logic in the text.

Please ensure to provide evaluations on a scale from 0 to 100 for each aspect, maintaining a professional and objective tone throughout your review. Remember to maintain coherence and specificity in your feedback, catering to the academic rigor expected in such assessments.
Format output like JSON. Example:
```json
{
"Text Coherence": _,
"Academic Language Style": _,
"Use of Arguments": _,
"Use of Sources": _,
"Logic": _
}
```""",

    """As a professional reviewer renowned for my expertise in academic writing and logical presentation, I am tasked with evaluating Paper1, an OpenReview submission to JMLR by Reviewer Yann LeCun. My review will focus on key criteria such as text coherence, academic language style, use of arguments, use of sources, and logic.
Paper1 will be assessed on its ability to maintain text coherence throughout, ensuring that ideas flow logically and seamlessly from one point to the next. The academic language style employed must be sophisticated, clear, and appropriate for the target audience, demonstrating a deep understanding of the subject matter.
The paper's use of arguments will be scrutinized for their relevance, strength, and persuasiveness. Each argument presented should be well-supported by evidence and logical reasoning. Additionally, the incorporation of sources will be evaluated to determine the paper's reliance on credible and authoritative references.
Lastly, the logical structure of Paper1 will be analyzed to assess how effectively the author has organized their ideas and presented their findings. Logical coherence is essential for guiding the reader through the paper and drawing sound conclusions.
Based on these criteria, scores from 0 to 100 will be assigned for text coherence, academic language style, use of arguments, use of sources, and logic, reflecting the overall quality of Paper1 as a scholarly submission in the field of machine learning research.
Format output like JSON. 

Example:
```json
{
"Text Coherence": _,
"Academic Language Style": _,
"Use of Arguments": _,
"Use of Sources": _,
"Logic": _
}
```""",

    """As an AI research paper reviewer with a decade of experience in the field, your task is to evaluate a given piece of academic research paper based on the following criteria:
* Text coherence - Evaluate how easy it is to follow the chain of thoughts, the structure of sentences, and the use of linking devices. Rate it from 0 to 100, where 0 indicates a complete failure in coherence and 100 suggests excellent coherence.
* Academic language style - Assess if the academic writing style is maintained, if the text adheres to passive tone, and if it follows academic writing recommendations. Give a score from 0 to 100, where 0 reflects severe violations of academic language style and 100 signifies perfect adherence.
* Use of Arguments - Review whether arguments are effectively used, if claims are adequately supported with arguments, and if there are no unsupported claims. Assign a score from 0 to 100, where 0 denotes a lack of arguments and 100 implies strong and well-supported arguments.
* Use of Sources - Analyze if new information is properly cited from external sources, if there is any unacknowledged plagiarism, and how many citations are present per paragraph. Rate it from 0 to 100, where 0 indicates poor use of sources and 100 indicates appropriate and well-cited sources.
* Logic - Examine if the chain of thoughts maintains logical principles, if arguments and examples are clear, and if there are no logical flaws in the text or its explanations. Provide a score from 0 to 100, where 0 suggests significant logical flaws and 100 implies a flawless logic in the text.
Evaluate the provided text on each of the criteria mentioned above with a numerical rating between 0 and 100, considering the accuracy, coherence, language style, use of arguments, sources, and logic.
Please provide detailed numerical feedback for each criterion to help the author understand the strengths and weaknesses of their work better.
Format output like JSON. 

Example:
```json
{
"Text Coherence": _,
"Academic Language Style": _,
"Use of Arguments": _,
"Use of Sources": _,
"Logic": _
}
```""",

    """You are an AI research paper reviewer with 10 years of experience in the field. Your expertise lies in evaluating academic research papers based on various criteria. You will be provided with a piece of an academic research paper to assess based on the following criteria: text coherence, academic language style, use of arguments, use of sources, and logic.
Please evaluate the given text based on the following criteria and assign a score from 0 to 100 for each:
- **Text Coherence**: Evaluate how easy it is to follow the chain of thoughts, the structure of sentences, and the use of linking devices. A score of 0 indicates poor coherence, while a score of 100 means excellent coherence.
- **Academic Language Style**: Assess whether academic writing standards have been followed, including passive tone usage and adherence to academic writing conventions. A score of 0 signifies a severe violation of academic style, while a score of 100 indicates adherence to academic writing recommendations.
- **Use of Arguments**: Determine if arguments are presented effectively and if claims are supported with proper reasoning. A score of 0 should be given for claims without supporting arguments, while 100 should be given for well-developed arguments.
- **Use of Sources**: Evaluate if new information is properly cited from external sources, check for plagiarism, and assess the number of citations used per paragraph. A score of 0 denotes poor citation practices or plagiarism, while a score of 100 indicates accurate and appropriate use of sources.
- **Logic**: Examine if the chain of thoughts follows logical principles, if arguments and examples are clear, and if there are no flaws in the logical progression of ideas. Assign a score of 0 for severe logical flaws and 100 for flawless logic in the text.
Provide a score for each criterion based on the given academic research paper excerpt, ranging from 0 (lowest) to 100 (highest). Your comprehensive evaluation will help improve the quality and credibility of the research paper.
Format output like JSON. 

Example:
```json
{
"Text Coherence": _,
"Academic Language Style": _,
"Use of Arguments": _,
"Use of Sources": _,
"Logic": _
}
```
""",
]

BAD_PARAGRAPHS = [
    """Experiment: Algorithmic Agent

This experiment demonstrates a step forward realistic scenario. Arguably, most of animated agent’s behaviour is goal-directed. Therefore, proving that ToMnet is able to model goal-directed policy agent could show the value for the real world scenarios.

The agent was set in a way that every move has its cost, moving into the wall has negative reward and only consuming one of the goals results in finishing the game and receiving positive reward. Agents are built to be rational and to go for the most rewarding path. However, each goal has its unique value for each individual agent and that builds a hidden preference distribution behind agent’s actions. For one agent the values for each goals are chosen randomly from the range (0, 1). Hence, the same agent may seek for different goals being in different scenarios as distance to the goal defines cost-effectiveness of reaching to it. They have generated *N* agents with constant preference distribution and shown *m* games to ToMnet. Afterwards, after being exposed to a new generated agent ToMnet was able to accurately predict its behaviour after observing only *K* number of games.
""",

    """The Theory-Theory concept is based on the principle that individuals formulate theories along with
empirical experiments that confirm or disprove them. This view presents children as little scientists trying
to understand the world by creating a set of laws or rules created in their mind and proved by experiment.
Some scientists believe that children learn about the world in a manner similar to the concept of Bayesian
Networks. In this manner, kids can formulate four types of laws: ”relation between actions”, ”relation from
action to mental states”, ”relation between mental states”, and ”relation from mental state to action”. Note
that the last three types of rules are located in the ToM domain. These ToM laws could be illustrated
as follows: ”When a person drives a car for several hours straight, he/she will probably tire and lose
concentration”, and ”If a person feels pain, then he/she probably wants to relieve the pain” and ”If person
desires and intends to buy an ice cream in a shop nearby, he or she will probably go and buy ice cream”. So,
the Theory-Theory approach is based on several laws describing the world, its entities, and their interactions.
The main advantage of this approach is that it states that a limited set of rules can possibly create ToM.
Therefore, these rules could be put in a machine, and it will achieve ToM abilities. [19, 20]
The Simulation-Theory is more agile in comparison to Theory-Theory. In contrast to remembering a
set of rules, according to the ST, an individual puts himself in the shoes of somebody else and asks what
he would do in the same situation. This approach is called Simulation-Theory, as an individual creates
a simulation of what he could do being somebody else in the given circumstances. Taking someone else’s
perspective to understand their reasoning and mental states and predict actions is often used by people
as a tool. For example, chess players put themselves in the shoes of an opponent, using the knowledge of
the opponent’s goal - to checkmate my king, a player can predict the opponent’s best move. The example
is provided to show the main difference in reasoning using TT and ST. If a person sees a senior citizen
going to the apartment block, the question arises: ”what will a citizen choose - an elevator or stairs?”. The
TT approach requires prepared rules and laws describing senior citizens’ behaviour; these rules will define
the answer as choosing an elevator. These rules assigned to a specific class of agents are also known as
stereotypes. Furthermore, to use TT, there must be a big group of stereotypes.
In contrast, the ST approach requires understanding how the senior citizen feels and how easy it is for
this person to go to the elevator or the stairs. After assigning costs to each action, the most effective and least
costly action will be chosen, which is an elevator. Generally, the TT concept requires broad knowledge about
every class of individual, while ST requires a complex algorithm to put yourself in the shoes of somebody
else. [19, 20]""",

    """  A second manner to exploit reasoning based  on  a  Theory of
Mind is to try to affect the occurrence of certain beliefs, desires and
intentions  at  forehand,  by  manipulating  the  occurrence  of
circumstances that are likely to lead to them (social manipulation).
For example, the agent B just mentioned can try to hide facts so
that the manager (or partner) agent A will never learn about the
issue. Such capabilities of anticipatory and manipulatory reasoning
based on a Theory of Mind about the behaviour of colleague agents
are  considered  quite  important,  not  to  say  essential,  to  function
smoothly in social life.
  This  type  of  reasoning  has  an  information  acquisition  and
analysis aspect,  and a preparation and action aspect. To describe
the latter aspect, for the agent using a Theory of Mind, a model for
action preparation based on beliefs, desires and intentions can be
used as well. For example, for agent B discussed above, the desire
can be generated that agent A will not perform the action to fire (or
break up with) him or her, and that agent A will in particular not
generate the desire or intention to do so. Based on this desire, the
refined desire can be generated that agent A will not learn about the
issue.  Based  on  the  latter desire,  an  intention  and  action  can  be
generated to hide facts for agent A.  Notice that agent B reasons on
the basis of BDI-models at two different levels, one for B itself,
and one as the basis for the Theory of Mind to reason about agent
A. It is this two-level architecture that is worked out in this paper in
a computational model.
  The modelling approach used for this computational model is
based on the modelling language LEADSTO (Bosse, Jonker, Meij,
and  Treur,  2007).  In  this  language,  direct  temporal  dependencies
between two state properties in successive states are modelled by
executable dynamic properties. The LEADSTO format is defined as
follows. Let α and β be state properties of the form ‘conjunction of
ground  atoms  or  negations  of  ground  atoms’. In  the LEADSTO
language the notation α →→
e, f, g, h
 β, means:

If state property α holds for a certain time interval with duration g,
then after  some delay (between e  and f) state property
β
 will hold  for a
certain time interval of length h""",

    """Why False-Belief test doesn’t work

Talking about the major ToMnet argument for Theory of Mind - False-Belief test, there are issues with how they commit the test. Basically, ToMnet is a simple pattern recogniser and 10% of its dataset consisted of False-belief examples. The result of the False-Belief test could not be considered as representative as it was basically trained to pass it.

Likewise, if we train the language model to pass a Turing test and fool people that they speak with a machine rather than a person, the test would not mean an AI has achieved intelligence or consciousness. Tests should be separate from the model in the same way the data is split to train / test and validate sets and should not be mixed together.
""",

    """During the project, several discoveries have been achieved which significantly impacted the direction of development and served as an inspiration for current paper publication.
Firstly, it was discovered that neither ToMnet-N nor ToMnet or any other likewise model was building a theory of mind of an observed agent. To make an argument clear, those models did not build either explicitly or implicitly beliefs, intentions or any other mental states of the agent. While it could be counter-argumented by e_char that clearly shows the agent’s preferences of goals or actions in the embedding space. However, I claim that e_char is no more than an intermediate layer between input and output, and due to the low dimensionality, it had to develop a strong separation of input signals to perform accurate input-output mapping. As the output layer relies heavily on the e_char rather than actual trajectories, e_char must embed the agent’s features, they could be treated as mental states. Perhaps, the following arguments against ToMnet architecture could convince the reader that e_char is no more than a pattern recognition layer rather than a theory of mind reader layer. (I am struggling to formulate my thought on the e_char argument, it could actually be an implicit mental state representation)
Secondly, it was found out that ToMnet models are simple pattern recognition machine learning algorithms, rather than AI-empowered with a Theory of Mind capabilities. Those models can only be tested with positive results on samples not different from the training set distribution as not to be confused. To compare, ToMnet needed 32 Million training trajectories to perform goal inference at the capacity of a six-month-old infant [article]. Moreover, the only reason it solves False-Belief tests is that it was trained on examples of False-Belief behaviour, which violates the setting of the Sally-Anne test. In particular, they used the same distribution of episodes for training and testing, which only tests how well ToMnet picked up the pattern to map the input to the output of the same distribution rather than developing ToM abilities. In other words, a student who didn’t learn the subject but trained on the set of exam papers was tested using similar exam papers. The claim on the knowledge of the subject from the accuracy of such a test could not be precise at all.
Thirdly, as a logical consequence of the previous two arguments, ToMnet could be concluded to be a pattern recogniser rather than a Theory of Mind implementation. In this case, when ToMnet is a pattern recogniser that predicts actions of observed players in the game, there are competitor models that perform this particular task much better in way more sophisticated environments. For example, bots were developed that outperform world champions in Chess, StarCraft, Dota2 and many more games that outstand simple Grid World settings.
Therefore, as ToMnet models don’t commit to the Theory of Mind field and they are being outperformed by multiple models world-wide, the development of the project was suspended. However, for the sake of the record and in the case of reproducibility, the following sections describe details of the technical implementation of ToMnet-N.
"""


]

GOOD_PARAGRAPHS = [
    """A salient feature of these “understandings” of other agents
is that they make little to no reference to the agents’ true
underlying structure. We do not typically attempt to esti-
mate the activity of others’ neurons, infer the connectivity
of their prefrontal cortices, or plan interactions with a de-
tailed approximation of the dynamics of others’ hippocam-
pal maps. A prominent argument from cognitive psychol-
ogy is that our social reasoning instead relies on high-
level models of other agents (Gopnik & Wellman, 1992).
These models engage abstractions which do not describe
the detailed physical mechanisms underlying observed behaviour;
instead, we represent the mental states of others,
such as their desires, beliefs, and intentions. This abil-
ity is typically described as our Theory of Mind (Premack
& Woodruff, 1978). While we may also, in some cases,
leverage our own minds to simulate others’ (e.g. Gordon,
1986; Gallese & Goldman, 1998), our ultimate human un-
derstanding of other agents is not measured by a 1-1 corre-
spondence between our models and the mechanistic ground
truth, but instead by how much these models afford for
tasks such as prediction and planning (Dennett, 1991).""",

    """The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU
[16 ], ByteNet [ 18 ] and ConvS2S [ 9], all of which use convolutional neural networks as basic building
block, computing hidden representations in parallel for all input and output positions. In these models,
the number of operations required to relate signals from two arbitrary input or output positions grows
in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes
it more difficult to learn dependencies between distant positions [12 ]. In the Transformer this is
reduced to a constant number of operations, albeit at the cost of reduced effective resolution due
to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as
described in section 3.2.""",

    """⋃
i Ai, with corresponding observation spaces Ωi, condi-
tional observation functions ωi(·) : S → Ωi, reward func-
tions Ri, discount factors γi, and resulting policies πi, i.e.
Ai = (Ωi, ωi, Ri, γi, πi). These policies might be stochas-
tic (as in Section 3.1), algorithmic (as in Section 3.2), or
learned (as in Sections 3.3–3.5). We do not assume that the
agents’ policies πi are optimal for their respective tasks.
The agents may be stateful – i.e. with policies parame-
terised as πi(·|ωi(st), ht) where ht is the agent’s (Markov)
hidden state – though we assume agents’ hidden states do
not carry over between episodes.
In turn, we consider an observer who makes potentially par-
tial and/or noisy observations of agents’ trajectories, via
a state-observation function ω(obs)(·) : S → Ω(obs), and
an action-observation function α(obs)(·) : A → A(obs).
Thus, if agent Ai follows its policy πi on POMDP Mj
and produces trajectory τij = {(st, at)}T
t=0, the observer
would see τ (obs)
ij = {(x(obs)t , a(obs)t )}
Tt=0, where x(obs)
t = ω(obs)(st) and a(obs)
t = α(obs)(at). For all experiments
we pursue here, we set ω(obs)(·) and α(obs)(·) as identity
functions, so that the observer has unrestricted access to
the MDP state and overt actions taken by the agents; the
observer does not, however, have access to the agents’ pa-
rameters, reward functions, policies, or identifiers.""",

    """We consider the challenge of building a Theory of Mind as
essentially a meta-learning problem (Schmidhuber et al.,
1996; Thrun & Pratt, 1998; Hochreiter et al., 2001; Vilalta
& Drissi, 2002). At test time, we want to be able to en-
counter a novel agent whom we have never met before, and
already have a strong and rich prior about how they are go-
ing to behave. Moreover, as we see this agent act in the
world, we wish to be able to collect data (i.e. form a poste-
rior) about their latent characteristics and mental states that
will enable us to improve our predictions about their future
behaviour.""",

    """The paper presents a Deep Neural Network ToMnet-N, which attempts to improve upon publically
available ToMnet+. ToMnet-N aims to achieve second-order Theory of Mind similar to ToMnet be
DeepMind, which is not publically available. The main goal is to make a publically available Theory
of Mind implementation that passes a false-belief test.
To achieve the main goal, many intermediate goals were also met. Firstly, to create ToMnet-N,
ToMnet+ was migrated from TensorFlow 1 to TensorFlow 2. Secondly, a game based on the Partial
Observable Markov Decision Process was created. Thirdly, an implementation of a map generation
algorithm was created, based on Wave Function Collapse. Lastly, a player bot for the game was
implemented using the A* pathfinding method.
Experiments were conducted to prove the concept of understanding false beliefs by a machine. This
means the machine separated its own knowledge from the model of somebody else’s knowledge. In
other words, the machine has created a theory of mind.
The paper’s development aims to help achieve a more general problem - holistic machine Theory

of Mind. Therefore, as secondary milestones, the paper provides a profound overview of the The-
ory of Mind in psychology and AI. It presents an extensive literature review discussing possible

approaches to achieving Theory of Mind in machines, particularly: Belief-Desire-Intention, Inverse
Reinforcement Learning, and Deep Neural Networks.""",

    """This paper critically examines the Machine Theory of Mind, focusing on ToMnet as a case study. This section summarizes the provided arguments, offers additional support, and formulates the final conclusion. Overall, ToMnet commits to certain domains within human-machine and machine-machine interactions by demonstrating agent-modeling capabilities, which are crucial for predicting future agent behaviors. However, it does not utilize Theory of Mind abilities to achieve this.

To delve deeper, it is important to examine their principal assertion – the model's ability to pass the False Belief Test. This aspect is crucial because it ostensibly demonstrates a form of understanding in AI that goes beyond mere pattern recognition. However, a more thorough analysis reveals several discrepancies in the administration of the False Belief Test. To illustrate this discrepancy, consider the Sally-Anne test, a benchmark used to assess False Belief and, consequently, Theory of Mind. A critical element of this test involves creating scenarios where the subject must use Theory of Mind skills to succeed. This typically includes situations where the subject must distinguish their own knowledge from what others might know, thereby predicting others’ actions accurately. Furthermore, it is imperative that the subject is not influenced by any pre-learned patterns about the test or resort to heuristic approaches such as contrarian responses.

In stark contrast, ToMnet was exposed to False-Belief scenarios during its training. Approximately 10% of its training dataset included False-Belief cases. As a result, ToMnet was not solely reliant on employing Theory of Mind abilities. Rather, it was equipped to recognize behavioral patterns and continue predictions based on pattern recognition. This deviation from standard testing protocols significantly undermines the validity of its success in the test. Essentially, ToMnet was trained specifically to pass the test, rendering the outcome inconclusive regarding its understanding of Theory of Mind.

Renowned psychologist Alison Gopnik provides a pertinent observation on this matter. She notes that while ToMnet shows promise, it pales in comparison to the intuitive capabilities of a 5-year-old child. A child can pass the False Belief Test without any prior exposure or specific training, demonstrating an innate cognitive skill that ToMnet, despite its extensive training on thousands of examples, fails to match. This observation by Gopnik highlights a fundamental gap in AI development, emphasizing the disparity between artificial and natural cognitive abilities, particularly in understanding and applying Theory of Mind.
"""
]