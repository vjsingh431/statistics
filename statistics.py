#!/usr/bin/env python
# coding: utf-8

# Q1. What is the Probability density function?
# Ans- A function that defines the relationship between a random variable and its probability, such that you can find the probability of the variable using the function, is called a Probability Density Function (PDF) in statistics.
Q2. What are the types of Probability distribution?
Ans- Types of Probability Distribution
There are two types of probability distribution which are used for different purposes and various types of the data generation process.

1)Normal or Cumulative Probability Distribution-The cumulative probability distribution is also known as a continuous probability distribution. In this distribution, the set of possible outcomes can take on values in a continuous range.For example, a set of real numbers.
2)Binomial or Discrete Probability Distribution- the normal distribution statistics estimates many natural events so well, it has evolved into a standard of recommendation for many probability queries. Some of the examples are:

Height of the Population of the world.
Rolling a dice (once or multiple times).
Tossing a coin.#Q3. Write a Python function to calculate the probability density function of a normal distribution with
given mean and standard deviation at a given point.
# In[3]:


#Ans3
from scipy.stats import norm 
import numpy as np 

data_start = -7
data_end = 7
data_points = 14
data = np.linspace(data_start, data_end, data_points) 

mean = np.mean(data) 
std = np.std(data) 

probability_pdf = norm.pdf(3, loc=mean, scale=std) 
print(probability_pdf)

Q4. What are the properties of Binomial distribution? Give two examples of events where binomial
distribution can be applied.
Ans-he binomial distribution is the probability distribution of a binomial random variable. A random variable is a real-valued function whose domain is the sample space of a random experiment.
The properties of the binomial distribution are:

There are only two distinct possible outcomes: true/false, success/failure, yes/no.
There is a fixed number of 'n' times repeated trials in a given experiment.
The probability of success or failure remains constant for each attempt/trial.
Only the successful attempts are calculated out of 'n' independent trials.
Every trial is an independent trial on its own, this means that the outcome of one trial has no effect on the outcome of another trial.
Example-Toss a fair coin twice. This is a binomial experiment. There are 4 possible outcomes of this experiment. {HH, HT, TH, TT}. Consider getting one head as the success. Count the number of successes in each possible outcome. 
No. of heads(n(X))	Probability of getting a head(P(X))
0	                     P(x = 0) = 1/4 = 0.25
1	                P(x = 1) = P(HT) = 1/4 + 1/4 = 0.50
2	                   P(x = 2) = P(HH) = 1/4 = 0.25

# Q5. Generate a random sample of size 1000 from a binomial distribution with probability of success 0.4
# and plot a histogram of the results using matplotlib.
# 
# 

# In[4]:


#Ans4
from scipy.stats import binom 
import matplotlib.pyplot as plt 
# setting the values 
# of n and p 
n = 1000
p = 0.4
# defining list of r values 
r_values = list(range(n + 1)) 
# list of pmf values 
dist = [binom.pmf(r, n, p) for r in r_values ] 
# plotting the graph  
plt.bar(r_values, dist) 
plt.show()


# #Q6. Write a Python function to calculate the cumulative distribution function of a Poisson distribution
# with given mean at a given point.

# In[7]:


#Ans-6Properties of CDF:

#Every cumulative distribution function F(X) is non-decreasing
#If maximum value of the cdf function is at x, F(x) = 1.
#The CDF ranges from 0 to 1.
# defining the libraries (mu= mean,n= size)
from scipy.stats import poisson
import matplotlib.pyplot as plt

#generate Poisson distribution with sample size 10000
x = poisson.rvs(mu=3, size=10000)

#create plot of Poisson distribution
plt.hist(x, density=True, edgecolor='black')

Q7. How Binomial distribution different from Poisson distribution?

Ans-Binomial Distribution:
1)It is biparametric, i.e. it has 2 parameters n and p.
2)he number of attempts are fixed.
3)There are only two possible outcomes-Success or failure.
4)Mean>Variance
Poisson distribution:
1)It is uniparametric, i.e. it has only 1 parameter m.
2)The number of attempts are unlimited.
3)There are unlimited possible outcomes.
4)Mean=Variance#Q8. Generate a random sample of size 1000 from a Poisson distribution with mean 5 and calculate thesample mean and variance.
from scipy.stats import poisson

#generate random values from Poisson distribution with mean=3 and sample size=1000
poisson.rvs(mu=5, size=1000)Q9. How mean and variance are related in Binomial distribution and Poisson distribution?
Ans-In binomial distribution Mean > Variance while in poisson distribution mean = variance.Q10. In normal distribution with respect to mean position, where does the least frequent data appear?
Ans-Diagram of Normal Distribution
This means that most of the observed data is clustered near the mean, while the data become less frequent when farther away from the mean. The resultant graph appears as bell-shaped where the mean, median, and mode are of the same values and appear at the peak of the curve.
# In[ ]:




