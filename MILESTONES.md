# Milestones

## 2025-02-17: Acquire Data 

### Technology Plan

Our data is stored is stored online using DoltHub, an online data storage platform, which you can pull to your local machine via Dolt, a dynamic very similar to GitHub and Git. There is also some additional data from CBOE. From there, there is some initial setup using Python scripts to get the data working on your machine. After that, Python and R will be used to analyze the data. We are still brainstorming what technology we want to use for dashboarding, but have been considering Shiny, Dash, and Streamlit.

### Project Goals

Our capstone project aims at utilizing research in basic quantitative finance and economic principles to add indicator variables to our stock and options data. We plan to then cluster companies based on attributes we determine to be of most relevancy using various unsupervised learning techniques. With our clusters and full data set, we will then incorporate an array of different machine learning algorithms trained on implied volatility to, at a minimum, predict historical volatility metrics. If the aforementioned goals are achieved, we plan to make near-future predictions as well.

### Data Wrangling

So far, our team has been focused on extracting data from the database, joining base data with additional data, and generating technical indicators. We have written scripts that let us take the data from our dolt database tables and save them into csv files. From here, data manipulation is made easier. We have a couple scripts we are working on that let us 1. adjust stock prices based on stock splits, 2. generate technical indicators, 3. generate some more naive volatility metrics, and 4. joining these stock prices with CBOE data. We have also been doing some basic data exploration to find limitations of the dataset that we are using. One big limitation we have found is that our options and volatility data is not reported at consistent intervals. There is some data reported weekly, some reported every couple of days, and some reported daily, and there are a couple holes through the dataset. We are working on writing scripts to clean this so we have weekly data with no holes across a multi-year time frame.

### Team Evaluation

So far, the team is making good progress towards our end-of-semester goal of having a functioning dashboard. While we have not yet begun the actual dashboarding or front-end development process, the work we have done so far preparing our massive dataset into something interpretable by data processing Python packages has us in a good place heading into week 5. In terms of obstacles, we need to be better at team communication and making consistent progress. We need to make sure that all team members are on the same page as we continue going forward hashing out the details of our plan, and to do this we will need consistent communication inside and outside of class.

As of now, our team is making steady progress toward achieving the end-of-semester goal of delivering a fully functioning dashboard. Although we still have a lot of work left to do we have made big strides in gathering and structuring the data necessary for our needs. However, some obstacles could hinder our ability to meet both weekly and end-of-semester goals. One challenge is determining which models we will need to use to answer different kinds of research questions about the data we are doing the dashboard on. Additionally, the size of the data could be another challenge as it takes time to process certain code due to that fact. To overcome these obstacles, we plan to continue communicating in a problem solving fashion that is being supportive and focused on the goal. We will continue to collaborate closely to address technical challenges early on to ensure the smooth development of the dashboard.
