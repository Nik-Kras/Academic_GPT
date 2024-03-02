# Academic_GPT
GPT for Academic Paper writing

This tool helps evaluating paragraphs from academic papers. It provides you with numerical metrics which you could use to build an optimal system that gets highest scores. On top of that, it has bit of explainability - the score is based on the averag of scores for each individual criteria. Check the output of paragraph evaluation and you will have a hint on what could you improve in hte paragraph: add more references? solve flaw in the logic-chain? improve your academic writing style?

It is a Methodology that you could apply to your dataset by changing prompts and data samples in `training_dataset.py`. In my case I have several examples of good and bad paragraphs and several prompts that may provide good evaluation. The methodology provided here analyses distributions of metrics for each feature and builds a statistical non-parametric model to predict if your paragraph is bad or good. That is why in the output you see a number 0-100, which is a value of a metric, following by a probability, that indicates a chance this paragraph is "good" based on statistical parameters model outputed during the training process.

Demo:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/YOUTUBE_VIDEO_ID_HERE/0.jpg)](https://www.youtube.com/watch?v=YbZfIN8qLKc)

