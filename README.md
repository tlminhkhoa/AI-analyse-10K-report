
AI analyse 10K :book: 
===
:star2: A machine learning approach to read financial reports 

* In our project, I will mine 10K reports by using BERTSUMTEXT  and FinBert to summarize  sentiment analysis important information.
* I also use a  random forest model to estimate the stock direction after the 10K report was announced.

![permalink setting demo](https://wsp-blog-images.s3.amazonaws.com/uploads/2021/12/09143736/Twitter-Annual-Report-vs-10K.jpg)


## Table of Contents

- [AI analyse 10K :book:](#ai-analyse-10k-book)
  * [Table of Contents](#table-of-contents)
  * [Backstory :film_projector:](#backstory-film_projector)
  * [Pipeline](#pipeline)
  * [Dataset](#dataset)
  * [Data Preprocessing](#data-preprocessing)
  * [Summarization (BERTSUMTEXT and Results)](#summarization-bertsumtext-and-results)
  * [Sentiment Analysis (Finbert and Results)](#sentiment-analysis-finbert-and-results)
- [Stock direction prediction :money_with_wings:](#stock-direction-prediction-money_with_wings)
  * [Data set](#data-set)
  * [Modeling (Random Forest)](#modeling-random-forest)
- [Limitations and possible extensions](#limitations-and-possible-extensions)


## Backstory :film_projector: 

The project come from a more personal problem. As I started learning how to invest, reading financial reports is a must. I was surprised by how long and detailed these reports are. Looking at the average length over year, it will be longer every year.
![](https://i.imgur.com/U9RZJoo.png)


As a data scientist who want to skim reading first, I asked myself 
> “Hmm, how machine learning can help me with this?”:exclamation: :exclamation: :exclamation:

As good programer, I googled the it first. And nothing was there. Sound like I have to make it for myself then. And I was harder than I expected :sweat_smile:

Pipeline
---
The following is my full data pipeline. I will go into more detail in the subsection.

![](https://i.imgur.com/nSkFBhJ.png)


Dataset
---
To achieve this, I built a tool to mine 10K reports from the S&P500 from 2016 to 2018 directly from the U.S. Securities and Exchange Commission (SEC) using the sec-edgar-downloader library.

The reason I choose the year range from 2016 to 2018 is to avoid the effect of Covid-19 on the financial market.

![](https://i.imgur.com/FNlfV10.jpg)

In total, I collected over 1400 financial reports, which total 70 GB of storage in the form of  HTML files. Please see the `Crawler.ipynb` for deatil code. 

For anyone trying to replicate the project, I do not encourage downloading all the datasets again. Follow the google drive links below to have access to my dataset

> Dataset: https://drive.google.com/drive/u/1/folders/17Txfj8Ceq_1MP016MwSmP6fhtN-zEnO_ 

> Reasonably, there should be 1500 total financial reports since there are 500 companies in 3 years, however, some are not accessible using the SEC API, or storge in the wrong year by the SEC database, these accounts for my missing documents. 


Data Preprocessing
---
I decided only to extract important sections from the report, namely Item1, Item1A, Item3, Item7, Item7A, and Item8. These items contain important information about a company from Business to Risk Factors and Legal Proceedings.
* Item 1. Business
* Item 1A.  Risk Factors
* Item 3.  Legal Proceedings
* Item 7. Management’s Discussion and Analysis of  Financial Condition and Results of Operations
* Item 7A. Quantitative and Qualitative Disclosures About Market Risk
* Item 8.Financial Statements and Supplementary Data

The text of the report is extracted using the index table and HTML internal links. The idea is to follow internal links from the HTML index table to get into the needed section. From there, we can extract text from the beginning of that section to its end.

Please see the `Processor.ipynb` file for more detail

The code design is intended for 10 colab sessions to run concurrently. To run the code, please modify the year to what year you want to process. Due to the large size of the dataset, I divide each year into 10 parts each, using multiple sessions by modifying the `worker` parameter.

> I also meet some problems in data processing, due to the lack of a standard format to structure the HTML reports. The SEC does require every 10k report to have the same sections, however, they do not regulate how these sections should be presented. I observed that different companies usually have different structures, even changing year over year, this is especially true for non-tech companies. For these reasons, only 710 examples remain.




Summarization (BERTSUMTEXT and Results)
---

BERTSUMEXT was pre-trained on CNN and DailyMail datasets, which contain news articles and associated highlights. 


On average, a report with 60,0146 words can be reduced down to 6,175, which is around 10% of the length.


<p align="center">
    <img
    width=""
    height="300"
    src="https://i.imgur.com/ISlBuVo.png">
        <img 
    width="300"
    height="300"
    src="https://encrypted-tbn3.gstatic.com/images?q=tbn:ANd9GcQ7NUH8gBQPcNuw2B8svHChlSaW38mXa83Jo4_AkibUbPVkS-fw">

</p>

The final output layer of BERTSUMEXT is a classifier which helps the model obtain the importance score for each sentence. The model ranks these sentences by their scores and selects the top-3 sentences as the summary. BERTSUMEXT achieved state-of-the-art performance on various datasets, but no study has applied it to annual reports.

<p align="center">
    <img 
    width="400"
    height="300"
    src="https://securecdn.pymnts.com/wp-content/uploads/2015/07/transparency-e1438278362553.jpg", style ="padding-right : 13px">
   
</p>


The reason I want to choose BERTSUMEXT over others methods is that BERTSUMEXT is able to produce important sentences from a given paragraph. I view this as a form of transparent feature extraction, which can remove unnecessary information and still be able to present human-level information. Transparency in finance is extremely important due to the million dollars behind every decision. Moreover, it also helps diagnose the system when needed.


---

BERTSUMEXT is able to detect and select important sentences given a paragraph. However, there is no gold standard for summarization tasks similar to this, so I decided to rely on a randomly chosen set of paragraphs to determine the effectiveness of the model. For the example below, the red color text is the model selected sentences, black is unchosen. 

![](https://i.imgur.com/3n7qgFJ.png)

As we can observe, the model correctly points out the marketing channel that Amazon acquires their customer and the reason why the company's marketing cost has increased.


---

![](https://i.imgur.com/yiRqWfY.png)

Here we can see that the first highlighted sentence talks about U.S dollar value and how it affects the company sales in general and since it is relevant it was chosen. But the sentence after it mentions belief and long-term goals which were not chosen in the summary.


---

![](https://i.imgur.com/nUj8Jva.png)

The main idea of the paragraph is that the source of revenue is the wide range of products. This was chosen while the other part which is a further explanation about the products and different types of sales was not.

> I observed the BERTSUMEXT model tends to choose the first sentence of the given paragraph, this is due to the model being trained on newspaper, of which the sentence is the headline or article title.


## Sentiment Analysis (Finbert and Results)

FINBERT is a BERT model pre-trained on financial communication text. It is trained on the following three financial communication corpus. Corporate Reports 10-K & 10-Q, Earnings Call Transcripts, and Analyst Reports. It is built by further training the BERT language model in the finance domain, using a large financial corpus, and thereby fine-tuning it for financial sentiment classification.


![image alt](https://content.presspage.com/uploads/2658/1920_headerimagetech7.png?10000)


In total, Finbert classifies 13.027 sentences as neutral, 1.289 sentences as positive, and 2.724 sentences as negative.  It is reasonable that most of the sentences are neutral since as the nature of the 10K, most of the material is reporting. 


However, the number of negative sentiments is double the number of positive ones. Interestingly, this is because the SEC required every company to have section 1A, “Risk Factor”, therefore, it introduces a dense negative section in each report.

<p align="center">
    <img
    width=""
    height="300"
    src="https://i.imgur.com/arUN4yb.png">
</p>

This fact can be observed when we break down the sentiment distribution. Most of the negative sentiment is section 1A, risk factor. The positive ones are distributed evenly and mostly around sections 1 and 7, which are “Business” and “Management’s Discussion and Analysis of  Financial Condition and Results of Operations” respectively.


<p align="center">
    <img
    width=""
    height="300"
    src="https://i.imgur.com/ALyyjBZ.png">
    <div style="text-align: center"> Positive sentiment word cloud </div>

</p>

For positive sentiments, words like “increase”, “new”, and “growth” occur more frequently in positive sentiment. These words usually indicate advantageous information for the company.


<p align="center">
    <img
    width=""
    height="300"
    src="https://i.imgur.com/uUhxPK7.png">
    <div style="text-align: center"> Negative sentiment word cloud </div>

</p>


Whereas the word indicates uncertainty like “may”,  or negative meaning like “adversely”, or “risk” occur more frequently in negative sentiment. Moreover, common words “market”, “oper”- root for operation, and “rate” indicate that our model successfully captures the context of the sentences not guessing based on words only.


<p align="center">
    <img
    width=""
    height="300"
    src="https://i.imgur.com/Yho2DeB.png">
    <div style="text-align: center"> Netral sentiment word cloud </div>

</p>


I randomly chose some paragraphs to investigate the model performance. The following are some examples.

![](https://i.imgur.com/lkP26jy.png)

The above sentence gets a classifying score of Neutral: -2.0949, Positive: -2.3267,  Negative: 6.3221
As the text mentions the company's limits in new markets and difficulties, the model correctly classifies it as negative.


As I mentioned before, 60% of sentiment is neutral. Below is an example of neutral sentiment.

![](https://i.imgur.com/arJqxBO.png)


This summarization is correctly selected as neutral since the sentences mention only the marketing channel without any implication of good or bad. 

Stock direction prediction :money_with_wings:
===

Normally due to the high noise-to-signal ratio of stock prices, it is not recommended to predict stock prices directly. The more sensitive approach is using quantitative analysis which neutralizes the market and measures the returns of the alpha factor for a portfolio in a stock universe.

 
However, I am personally curious about how a tree-based model can give insight into what section is the most important in a given report. This can give an investor knowledge of important sections that affect the stock return and should be paid attention to. 


Data set 
---

I also collected a 2-month stock return before and after the 10K announcement, usually from the beginning of December to the end of January next year. However, the company can delay its report, in that case, the months are adjusted. This is done by the Yahoo Finance API.

![](https://miro.medium.com/max/1400/1*PyZ91jfRlllJrCG7X4cCIA.jpeg)

The range is selected to remove speculation from the market. This is a common phenomenon where the market tries to guess the report’s content and influences price irrationally. Moreover, I also want to avoid the panic from the market that leads to overselling or overbuying after the report announces, choosing one month after to help the stock price return to rational. Therefore the 2 months range is chosen as a buffer for these 2 phenomena.



I decide to predict the stock price direction rather than the stock price itself due to the high noise-to-signal ratio of the stock price. It is common knowledge that stock price returns are distributed normally with fat tails, this means more extreme events are likely to occur. 


<p align="center">
    <img
    width=""
    height="300"
    src="https://i.imgur.com/88Eco1G.png">
    <div style="text-align: center">  2 months stock return collected </div> </p>

The above is my 2 months stock return from the stock universe, we can observe that although most of the value centers around 0, many extreme cases with loss or gain up to 300 still happen.I believe that we do not have enough data to cover all the extreme cases in regression. By converting to a classification problem, the model will cover these cases' directions correctly without specifying the value. Moreover, predicting the direction alone can bring large benefits to investors.  

This done by take the difference between the latter month to the earlier month to get the direction of the stock. Going down is 0, and Going up is 1.


<p align="center">
    <img
    width=""
    height="300"
    src="https://i.imgur.com/YYDdQKG.png">
    <div style="text-align: center"> Class distribution after transformation </div></p>


It can be observed that when stock returns are mapped into stock return direction, the label increase occurs more than decrease. This reflects the market trend of going up on average. 


Modeling (Random Forest)
---

To predict the stock direction, I employed a random forest model for the task.  The reason is the model's ability to deal with noise by bootstrapping  However, the risk of overfitting is still present due to the nature of the data.

Therefore 10 fold cross-validation is also used to gain model insight. I decided to choose 10-fold cross-validation over the train-test split method because of the lack of training data. 


To account for the imbalance of labels, where the label increase is more than the decrease, using the sklearn class_weight parameter, I put more weight on the decrease class, specifically 1.5 to 1. Another hyperparameter is tuned using grid search, which gives the following set:
* random_state=0
* max_depth =10
* n_estimators= 30


<p align="center">
    <img
    width=""
    height="300"
    src="https://i.imgur.com/QgOK8SM.png">
</p>

Due to the high noise-to-signal ratio, I achieve around 63% of accuracy, some can view this as low compared to other problems, however, stock direction prediction is reasonable. Much of the research I found, took [“Predicting Stock Price Direction using Support Vector Machines “](https://https://www.cs.princeton.edu/sites/default/files/uploads/saahil_madge.pdf) for example, only achieve 55% to 60% on test prediction. Therefore, the result can be viewed as acceptable. Our standard deviation is around 4%, indicating that the expected will not deviate far from the mean.

<p align="center">
    <img
    width=""
    height="500"
    src="https://i.imgur.com/DSJxvqT.png">
</p>

The feature name is mapped into numbers, the notion as follows:
* From 0 to 3 is the sentiment from section 1
* From 4 to 7 is the sentiment from section 1A
* From 8 to 11 is the sentiment from section 3
* From 12 to 15 is the sentiment from section 7
* From 16 to 19 is the sentiment from section 7A
* From 20 to 23 is the sentiment from section 8

According to our model, on average, sentiment from section 7 “Management’s Discussion and Analysis of  Financial Condition and Results of Operations” is the most important section to indicate a stock direction, followed by section 1 “Business”. Section 8 “Financial Statements and Supplementary Data” is the least informative. This suggests that investors should pay more attention to these sections than others.


Limitations and possible extensions
===

1. I still face some limitations related to raw data format. Most of the 10K I collected is unusable due to HTML structure. Given a cleaner set, more data can be extracted. 
2. The models I used are trained on news data which can be different from the data used in financial reports. Therefore the second limitation is a purely manual task because there are no summaries for annual reports available. Similarly, for sentiment, it requires generating labeled data. Alternatively, unsupervised summarization models could be examined.
3.  BERTSUMEXT sometimes leans toward choosing the first sentence of a given paragraph. The reason is because of what it was trained on (news-related data).
4.  Finally, the stock direction is measured only by the difference in price from 2 months. In a more quantitative analysis environment, this should be done by taking into account the market trend, removing risk factors, and measuring the alpha effect.


###### tags: `10K report` `Documentation` `machine learning` `Stock`
