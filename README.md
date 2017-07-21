# Machine Learning / Deep Learning Resources

A curated list of resources for machine learning and deep learning that I found useful.

Last updated: 7/2017

## Books
* [The Elements of Statistical Learning](http://web.stanford.edu/~hastie/ElemStatLearn/): 
Book by the authors of LASSO, gradient boosting, MARS, etc. Machine learning from a statistical viewpoint. Free pdf available online.
* [An Introduction to Statistical Learning](http://www-bcf.usc.edu/%7Egareth/ISL/):
An introductory version of the above book, by mostly the same authors.
* [Data Analysis Using Regression and Multilevel/Hierarchical Models](http://www.stat.columbia.edu/~gelman/arm/):
Introductory book by Andrew Gelman. Focuses on data analysis, inference, and explanation of your data using linear regression models. Very accessible.
* [Machine Learning: A Probabilistic Perspective](https://www.cs.ubc.ca/~murphyk/MLbook/):
Machine learning from a mostly Bayesian viewpoint. Covers a very wide range of ML algorithms in depth.
* [Pattern Recognition and Machine Learning](http://www.springer.com/us/book/9780387310732):
Classic Bayesian ML textbook.
* [Bayesian Data Analysis](http://www.stat.columbia.edu/~gelman/book/):
Bible of Bayesian analysis by Andrew Gelman.
* [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/):
Bible of convex optimization by Stephen Boyd. Free pdf available.
* [Deep Learning Book](http://www.deeplearningbook.org/):
Written by top deep learning researchers. Includes practical guidelines, theoretical justifications and advanced materials on recent research. Does not cover deep reinforcement learning. Free pdf available.

## Courses and Tutorials
* [Machine Learning (Coursera)](https://www.coursera.org/learn/machine-learning):
Most well-known intro ML online course by Andrew Ng. Very accessible and light in math.
* [Stanford CS229 - Machine Learning](http://cs229.stanford.edu/) 
([video](https://www.youtube.com/view_play_list?p=A89DCFA6ADACE599)):
Andrew Ng's course at Stanford; covers the deeper math and theory of ML. Handouts available on the website.
* [Stanford CS246 - Mining Massive Datasets](http://web.stanford.edu/class/cs246/)
([video](https://www.youtube.com/channel/UC_Oao2FYkLAUlUVkBfze4jg/videos),
[book](http://www.mmds.org/)):
Practical data mining methods, such as map-reduce, page-rank and recommendation system. Handouts available on the website.
* [Stanford Statistical Learning](http://online.stanford.edu/course/statistical-learning-self-paced):
Course based on the "Introduction to Statistical Learning" book, taught by its author.
* [Stanford Convex Optimization](http://stanford.edu/class/ee364a/index.html) 
([video](https://www.youtube.com/playlist?list=PL3940DD956CDF0622)):
Taught by Stephen Boyd himself.
* [Sctkit-learn Documentation](http://scikit-learn.org/stable/):
Very comprehensive documentation, covers many common ML algorithms and has a lot of practical examples.
### Deep Learning Materials
* [Stanford CS231n - Convolutional Network for Computer Vision](http://cs231n.stanford.edu/)
([video](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC)): By Fei-Fei Li and Andrej Karpathy. Deep learning basics, convolutional net and computer vision applications
* [Stanford CS224n - NLP with Deep Learning](http://cs224n.stanford.edu/)
([video](https://www.youtube.com/playlist?list=PL3FW7Lu3i5Jsnh1rnUwq_TcylNr7EkRe6)):
By Chris Manning and Richard Socher. Word2vec, recurrent network, machine translation, and other NLP applications.
* [Berkeley CS294 - Deep Reinforcement Learning](http://rll.berkeley.edu/deeprlcourse/)
([video](https://www.youtube.com/playlist?list=PLkFD6_40KJIwTmSbCv9OVJB3YaO4sFwkX))
* [Neural Networks (Coursera)](https://www.coursera.org/learn/neural-networks):
By Geoffrey Hinton, godfather of the modern deep learning.
* [Oxford Machine Learning](http://www.cs.ox.ac.uk/teaching/courses/2014-2015/ml/) 
([video](https://www.youtube.com/playlist?list=PLE6Wd9FR--EfW8dtjAuPoTuPcqmOV53Fu)):
By Nando de Freitas. Starts from basic ML and dives into deep learning.
* [Neural networks - Université de Sherbrooke](http://info.usherbrooke.ca/hlarochelle/neural_networks/content.html)
([video](https://www.youtube.com/playlist?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)): By Hugo Larochelle.
* [2016 Deep Learning Summer School at Montreal](https://www.youtube.com/playlist?list=PL5bqIc6XopCbb-FvnHmD1neVlQKwGzQyR)
* [Deep Learning Tutorials](http://deeplearning.net/tutorial/): Tutorial from Yoshua Bengio's prior course
* [Unsupervised Feature Learning and Deep Learning](http://deeplearning.stanford.edu/tutorial/):
Tutorial from Andrew Ng's prior deep learning course.
* [Tensorflow Tutorial](https://www.tensorflow.org/tutorials/)

## Research Papers
* [Most Cited Deep Learning Papers](https://github.com/terryum/awesome-deep-learning-papers):
A list of important deep learning papers.
* [Deep Learning Papers Reading Roadmap](https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap):
Another good list of deep learning papers.
* [Arxiv Sanity](http://www.arxiv-sanity.com/): Andrej Karpathy's tool to help you find good papers on arxiv.
* [Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/):
Actually a place where people discuss good recent research. Paper authors sometimes respond to comments as well.

## Useful Libraries (Mostly Python)
* [numpy + scipy](https://scipy.org/): Fast vector and matrix operations, linear algebra, optimization, sparse matrix.
* [scikit-learn](http://scikit-learn.org/): Most popular ML library in Python.
* [pandas](http://pandas.pydata.org/): Data wrangling and analysis.
* [statsmodel](http://www.statsmodels.org): Statistics functions. 
* [cvxpy](http://www.cvxpy.org/): Convex optimization.
* [stan](http://mc-stan.org/) and [pymc](http://pymc-devs.github.io/pymc3/): Bayesian modeling and inferences.
* [opencv](http://opencv.org/) and [scikit-image](http://scikit-image.org/): Computer vision and image analysis.
* [nltk](http://www.nltk.org/) and [spacy](https://spacy.io/): Natural laguage processing. Spacy is newer and more performant.
### Gradient Boosting Machine (GBM)
* [xgboost](http://xgboost.readthedocs.io/en/latest/): Most popular and well-tested GBM package.
* [lightgbm](https://github.com/Microsoft/LightGBM): A newer library by Microsoft. 5-10X faster than xgboost default mode.
* [catboost](https://github.com/catboost/catboost): Another new libary by Yandex. Handles categorial features naturally and claims to be more accurate than prior libraries.
### Deep Learning
* [theano](http://deeplearning.net/software/theano/): One of the early deep learning libary widely used in academia.
* [torch](http://torch.ch/): Another early library popular in academia. It is in Lua instead of Python.
* [caffe](http://caffe.berkeleyvision.org/): Popular library for conv net. Has a lot of pretrained models.
* [tensorflow](https://www.tensorflow.org/): Backed by Google, arguablly the most popular libary now. API is quite similar to theano.
* [caffe2](https://caffe2.ai/): Successor of caffe by Facebook.
* [pytorch](http://pytorch.org/): Bring torch to Python, also by Facebook.
* [mxnet](http://mxnet.io/): An open-sourced framework (Apache incubator) backed by Amazon
* [cntk](https://www.microsoft.com/en-us/cognitive-toolkit/): Deep learning framework by Microsoft.
* [keras](https://keras.io/): Provides high-level deep learning API that runs on the top of Tensorflow, theano or CNTK. Very user friendly. Now officially supported in tensorflow.
