# Milestones

## 2025-02-17: Acquire Data Milestone

### Technology Plan

Our data is stored is stored online using DoltHub, an online data storage platform, which you can pull to your local machine via Dolt, a dynamic very similar to GitHub and Git. There is also some additional data from CBOE. From there, there is some initial setup using Python scripts to get the data working on your machine. After that, Python and R will be used to analyze the data. We are still brainstorming what technology we want to use for dashboarding, but have been considering Shiny, Dash, and Streamlit.

### Project Goals

Our capstone project aims at utilizing research in basic quantitative finance and economic principles to add indicator variables to our stock and options data. We plan to then cluster companies based on attributes we determine to be of most relevancy using various unsupervised learning techniques. With our clusters and full data set, we will then incorporate an array of different machine learning algorithms trained on implied volatility to, at a minimum, predict historical volatility metrics. If the aforementioned goals are achieved, we plan to make near-future predictions as well.

### Data Wrangling

So far, our team has been focused on extracting data from the database, joining base data with additional data, and generating technical indicators. We have written scripts that let us take the data from our dolt database tables and save them into csv files. From here, data manipulation is made easier. We have a couple scripts we are working on that let us 1. adjust stock prices based on stock splits, 2. generate technical indicators, 3. generate some more naive volatility metrics, and 4. joining these stock prices with CBOE data. We have also been doing some basic data exploration to find limitations of the dataset that we are using. One big limitation we have found is that our options and volatility data is not reported at consistent intervals. There is some data reported weekly, some reported every couple of days, and some reported daily, and there are a couple holes through the dataset. We are working on writing scripts to clean this so we have weekly data with no holes across a multi-year time frame.

### Team Evaluation

So far, the team is making good progress towards our end-of-semester goal of having a functioning dashboard. While we have not yet begun the actual dashboarding or front-end development process, the work we have done so far preparing our massive dataset into something interpretable by data processing Python packages has us in a good place heading into week 5. In terms of obstacles, we need to be better at team communication and making consistent progress. We need to make sure that all team members are on the same page as we continue going forward hashing out the details of our plan, and to do this we will need consistent communication inside and outside of class.

As of now, our team is making steady progress toward achieving the end-of-semester goal of delivering a fully functioning dashboard. Although we still have a lot of work left to do we have made big strides in gathering and structuring the data necessary for our needs. However, some obstacles could hinder our ability to meet both weekly and end-of-semester goals. One challenge is determining which models we will need to use to answer different kinds of research questions about the data we are doing the dashboard on. Additionally, the size of the data could be another challenge as it takes time to process certain code due to that fact. To overcome these obstacles, we plan to continue communicating in a problem solving fashion that is being supportive and focused on the goal. We will continue to collaborate closely to address technical challenges early on to ensure the smooth development of the dashboard.

## 2025-02-23: Project Goals Milestone

### Exploratory Analysis

We have been using the Polars library of Python for our data analysis, using it to turn our .csv files into dataframes and manipulating them with Python code. We've been making some initial exploratory plots in using MatPlot PyPlot. Our options data, which included the fields we intended to use as response variables, does not have data reported at consistent rates, so we will have to do some manipulation and use only weekly data here. So far, our exploration has involved generating and analyzing different technical indicators to see which ones are useful and which ones are redundant. We are narrowing in on which indicators will be useful for modeling volatility and which we can scrap. We have done some really basic visualizations while working on our exploration, but nothing that will end up in the final dashboard yet.

### Modeling Plan

For modeling, right now we are focused on doing further exploration of our variables and understanding what will actually go into our models, which will let us narrow down what model options work with our data. Our main modeling task involves stock data and technical indicators, as well as some realized volatility metrics, to model implied volatility and lagged realized volatility. We have also been considering utilizing some unsupervised learning models that can cluster our stocks, then we could create unique models for each cluster, so different categories of stocks can utilize slightly different models. Currently, we are working on finalizing our technical indicators and creating some of our more naive realized volatility metrics. There are industry standard models we could use for our initial volatility metrics, such as GARCH models, which may work well but wouldn't scale to match our project, so we're probably going to stick with simpler metrics like Jefferson's volatility and Yang-Zhang volatility. For our main modeling task, we are considering models like LSTM, LightGBM, and XGBoost, which would utilize the implied volatility or lagged realized volatility as response variables, and we could use our stock data and indicators as explanatory variables. We need to make sure that we aren't double dipping into our data and using adequate train and test splits. If we use a more advanced model like LSTM or Prophet, we will also need to do some more research to understand what retraining or finetuning is necessary.

### Project Goals

Our target audience includes students, researchers, and investors who are interested in learning about what factors impact stock volatility and in which areas predicting volatility can be important for pricing derivatives. We will continue to research what kind of models would work best to be used within a dashboard. We haven't started dashboarding yet, but as we implement some naive volatility metrics, we can begin creating different views for our dashboard. We have also been exploring other dashboarding projects out on the internet which involve futures and portfolio management, and gathering ideas on what might be good to implement into our dashboard. Some of the details are subject to change as we complete more research, but overall, the next few weeks will be focused on selecting technical indicators, volatility metrics, and mapping out our modeling procedures.

### Project Progress

So far, each team member has contributed to research on the project topic. Not everybody has committed changes to GitHub, as some of the team is still learning about how version control systems. Everyone on the team has their local tech stack ready and has confirmed that the scripts in the repo run correctly on their local systems. The next steps for the project are to pick appropriate models and finish finding the most important technical indicators. Individually, team members are currently split between focusing on technical indicator research, improving existing scripts, and exploring naive volatility metrics that can be used for our models. For communication, we have been discussing progress updates and project trajectory in the Discord, and all team members have been communicative. Moving forward, each team member has a clear plan for their next steps, and we are all on the same page about next steps for the project.

## 2025-03-02: Exploratory Analysis Milestone

### Brainstorm Dashboard

Our dashboard will be aimed at providing interactive insights into volatility of different stocks. The user should be able to choose a stock and a date (or date range), and have a few different ways to visualize our modeled volatility for that period. We plan on adding features so that the user is able to look at historical volatility and compare to our modeled volatilities. We want users to be able to see what factors impact volatility, and then see how volatility impacts things like derivative pricing, portfolio risk management, and statistical arbitrage opportunities. Past this, we have a few dashboards already that we are planning on using as inspiration and will continue to brainstorm the intricacies of the dashboard as our project materializes in the next week or two.

### Data Report

Our dataset contains time series data for over 2000 equities, including daily open, high, low, and close prices, and volume. This dates back to 2011. The database we get this from also has some information on derivatives and stock splits on the same tickers. We have a second dataset which contains some CBOE volatility metrics, also reported daily since 2011 (some of the metrics are reported earlier as well). Finally, we have a third dataset containing option chain and volatility data for over 2000 equities reported weekly since 2019. We have lots of data, but by using Polars and doing clever data engineering techniques, it's definitely manageable. We are using the ohlcv data, and we start by applying the split ratios found in the stock splits table so that our prices and volume are adjusted appropriately. Then, we generate different technical indicators and volatility metrics using polars on the ohlcv data. We finish by joining this to the different cboe data csvs, so that the daily volatility metric data is appended to our dataframe. This data should be enough for our applications. We are currently exploring whether or not we will really need to use the options side of the data (discussed further in Exploratory Analysis) since the tables there are massive compared to the rest of the datasets. 

### Project Progress

So far, the team has been focused on finalizing features for modeling. We have been building out technical indicators and volatility metrics from our base data, and cleaning and consolidating data where needed. We have started on exploratory analysis and looking at feature selection techniques for further narrowing down and refining our feature set. The project has been very research-heavy so far, with different team members exploring the math and industry knowledge that has driven development of the technical indicators we are using, but we are starting to head in a more standard visualization and exploration direction. For next steps, we are considering feature selection and clustering approaches, trying to decide what models are most applicable for our data, creating extensive documentation of our generated features, and making a decision on what our response variable will be (we have a couple different options including historical variance and implied volatility, and we might even use a generated volatility metric as our response variable). Once we finalize these decisions, the modeling task becomes tangible. The team is communicating well, it has been a slow week since we all had exams and other class work due this week, but we are ready for a big push this week.

### Exploratory Analysis

Our dataset contains time series data for over 2000 equities, including daily open, high, low, and close prices, and volume. This dates back to 2011. The database we get this from also has some information on derivatives and stock splits on the same tickers. We have a second dataset which contains some CBOE volatility metrics, also reported daily since 2011 (some of the metrics are reported earlier as well). Finally, we have a third dataset containing option chain and volatility data for over 2000 equities reported weekly since 2019. One issue that we have found with our data is that some equities have been removed from the market between the dataset's creation and now, while others have been added to the market after the dataset's creation. We will likely have to remove both of these kinds of data, as otherwise we will have lots of 'N/A' values. In terms of important features that we are finding, we have been looking into different volatility metrics, and have found that Yang-Zhang volatility provides results very close to the historical variance data derived from options prices. If we are able to get a volatility metric working well enough, we could simply lag this metric and use it as our response variable, letting us utilize the daily data for training instead of relying on the weekly data for our response variable. We are still exploring volatility metrics and determining if this is really a good idea or not.

[![Chart showing different volatility metrics for AAPL data over 2020](docs/MS20250302EA.png)](docs/MS20250302EA.png)

## 2025-03-09: Brainstorm Dashboard Milestone

### Finalized ETL Pipeline (Team Determined Mini-Milestone)

For our team determined mini-milestone this week, we wanted to finalize and standardize our data processing pipeline such that users could easily load in our data and get it to a 'model-ready' state. For this, we have a series of files in the [src/data_transformation](src/data_transformation) folder. After the initial database setup and extraction, users can collect their ohlcv csv into a Polars dataframe using this command:

```{python}
ohlcv = pl.read_csv("data/raw/stocks/csv/ohlcv.csv").with_columns(
    pl.col("date").str.to_date("%Y-%m-%d")
)
```

And then apply feature transformations using this command:

```{python}
ohlcv = src.transformation_pipeline.transformation_pipeline(ohlcv)
```

It will take around a minute to run. First, it runs the `remove_incomplete_tickers` and `adjust_splits` methods from [src/data_transformation/stock_adjustments.py](src/data_transformation/stock_adjustments.py), then it'll generate technical indicators using `add_technical_indicators` method from [src/data_transformation/technical_indicators.py](src/data_transformation/technical_indicators.py), and finally joins our data with the cboe volatility index csvs using the `join_cboe_indices` method from [src/data_transformation/cboe_index_join.py](src/data_transformation/cboe_index_join.py). This setup is modular, and allows us to add technical indicators or further stock split adjustments as needed while maintaining ease of use for developers.

### Brainstorm Dashboard

We are going to make our dashboard showcase different volatility metrics including our modeled volatility predictions for the various stocks in our dataset. One of our big desires for this dashboard is a high degree of interactivity, where users are able to adjust the graphs contained within to their liking, such as picking specific stocks and date ranges, to make the tool as versatile as possible. We want our users to be able to use our dashboard to model historical volatility and also use it to look a little bit into the future to predict future volatility, as well as showing our modeled historical volatility vs realized historical volatility. We also want to include some other metrics to inspire confidence in our volatility predictions and hopefully show how our models came to their conclusions. From there, we want to be able to provide users with some additional stats that are taken from our volatility predictions, the goal being to give any users of our dashboard an overview of the modeled stocks and any stats or metrics that are of interest to traders, to give them a detailed profile of the selected stocks if so desired.

### Finalize Data Models

Our final models have a few main components. First, we need to use a dimensionality reduction technique like PCA to determine which indicators/attributes would be best to cluster on. From here, we will use some sort of unsupervised learning technique, which then gives us clusters that we will make our final models customized on. From our testing, we aren't entirely sure if this clustering approach will be appropriate, but we still have a few options left to try. Because our data is of a time-series nature and large, we have a few things on our final model wishlist. Ideally, our model will not require normalization of data, allowing us to side-step large amounts of computation time (normalization is harder when we have highly temporal data). Additionally, we would like our model to be equipped with the ability to handle temporal data, either out-of-box or with simple modifications, which is something that not all models can do. We need this model to be efficient with large data and customizable to each cluster, as we would ideally want to build out a separate model for each cluster, but we will test some options that operate on the entire ticker pool as well. We have narrowed down our selection to XGBoost, Transformer Based Models, GRU Models, or some other LSTM model. From earlier research, we have concluded that we will be training our final model using our technical indicators and volatility metrics (both in-house and CBOE) as explanatory variables, with future historical volatility (i.e. historical volatility shifted forward) as the response variable.

### Project Progress

This week, each team member contributed to the implementation and research of different modeling, feature selection, and clustering approaches to decide which options suit our data and our needs the best. Each member has researched and gathered information about different approaches, and we have also been working on some basic implementation of found strategies. Every team member has their tech stack ready on their local machine, but there is still some friction with getting used to version control systems and the new technologies involved. The next steps for the project are to find more dashboard ideas and finalize our modeling approach. So far the team has been communicating well and it seems like everyone is on the same page for next steps. We have been working hard to make sure everyone understands the project and what direction we need to be going in to get our models and dashboard built.

## 2025-03-16: Finalize Data Models Milestone

### Finalize Data Models

We are still working on deciding exactly which data models we will use. We are starting to lean towards NN-based approaches, and have built out code for DeepAR models, while we have in-progress code for TFT and NBEATS approaches. We have already seen better results with DeepAR approaches as opposed to models like XGBoost which don't act on the temporal elements of our data. Utilizing models built for time-series with LSTM layers should give us the best possible results. Since we already have some results for XGBoost and DeepAR models, we can utilize those to build our dashboards and just 'plug in' a different model once we find the best solution. We are also considering trying to use an approach that can predict across multiple different time horizons and even build our an entire volatility surface instead of just one point. 

### Project Progress

This week, we have been focusing on generating dashboard ideas and finalizing our model selection. The team has been split across model research, implementation, dashboard generation, and dashboard research, but we are starting to collect our ideas into a more cohesive plan as we head into break. We have registered our dashboard as a streamlit app and we have been training some models to base our visualizations and predictions off of. We have also slightly shifted our prediction goals and have been doing research on models related to volatility surface predictions as opposed to point volatility predictions.

### Dashboard Sketch

We have started to explore how to best visualize our model results on our dashboard. We are planning to use Streamlit and it's associated Python packages to drive and host our dashboard. We're working on building out the workflow to get our model results and data out and displayed on Streamlit, but have been running into issues with our data size. Since we have very large datasets and models, we can't store the models on GitHub, which is where Streamlit reads our data from. We will have to explore more creative file storage and delivery solutions. Streamlit is a really expansive and powerful dashboarding tool, so we have lots of options for user interaction and visualization choices for our model results. We'd love to do something related to derivative pricing, something related to portfolio management, and/or something related to statistical arbitrage. We're also going to have multiple pages on our dashboard, which Streamlit should be able to accommodate. 

### Spring Break Plans

The group has decided to continue working on our project when possible during Spring Break. Discussions were held the day before break to determine which goals we would like to accomplish over break. We want to have a large push towards finalizing the model and beginning the dashboard. Ideally, we will have the model complete and the dashboard began by the end of break. We understand that each team member has different amounts of time they are able to commit, but will do what they can to take advantage of the time before next class period.
