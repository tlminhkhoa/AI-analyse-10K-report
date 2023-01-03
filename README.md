
AI analyse 10K :book: 
===
:star2: A machine learning approach to read financial reports 

* In our project, I will mine 10K reports by using BERTSUMTEXT  and FinBert to summarize  sentiment analysis important information.
* I also use a  random forest model to estimate the stock direction after the 10K report was announced.

![permalink setting demo](https://wsp-blog-images.s3.amazonaws.com/uploads/2021/12/09143736/Twitter-Annual-Report-vs-10K.jpg)


## Table of Contents

[TOC]

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
:::info 
In total, I collected over 1400 financial reports, which total 70 GB of storage in the form of  HTML files. Please see the `Crawler.ipynb` for deatil code. 
:::
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

:::info 
On average, a report with 60,0146 words can be reduced down to 6,175, which is around 10% of the length.
:::

<p style="text-align: center">
    <img
    width=""
    height="300"
    src="https://i.imgur.com/ISlBuVo.png">
        <img 
    width="300"
    height="300"
    src="https://storage.googleapis.com/kaggle-datasets-images/1654566/2715323/44ef1513e9d7d4f78cbe1aeee8c1a866/dataset-card.jpg?t=2021-10-17-19-36-14?t=2021-10-18-05-20-13">

</p>

The final output layer of BERTSUMEXT is a classifier which helps the model obtain the importance score for each sentence. The model ranks these sentences by their scores and selects the top-3 sentences as the summary. BERTSUMEXT achieved state-of-the-art performance on various datasets, but no study has applied it to annual reports.

<p style="display: flex; float: left; ">
    <img 
    width="400"
    height="300"
    src="https://securecdn.pymnts.com/wp-content/uploads/2015/07/transparency-e1438278362553.jpg", style ="padding-right : 13px">
   
<div style ="padding-bottom: 15px" =>
      The reason I want to choose BERTSUMEXT over others methods is that BERTSUMEXT is able to produce important sentences from a given paragraph. I view this as a form of transparent feature extraction, which can remove unnecessary information and still be able to present human-level information. Transparency in finance is extremely important due to the million dollars behind every decision. Moreover, it also helps diagnose the system when needed.</div>
</p>




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

:::info 
In total, Finbert classifies 13.027 sentences as neutral, 1.289 sentences as positive, and 2.724 sentences as negative.  It is reasonable that most of the sentences are neutral since as the nature of the 10K, most of the material is reporting. 
:::

However, the number of negative sentiments is double the number of positive ones. Interestingly, this is because the SEC required every company to have section 1A, “Risk Factor”, therefore, it introduces a dense negative section in each report.

<p style="text-align: center">
    <img
    width=""
    height="300"
    src="https://i.imgur.com/arUN4yb.png">
</p>

This fact can be observed when we break down the sentiment distribution. Most of the negative sentiment is section 1A, risk factor. The positive ones are distributed evenly and mostly around sections 1 and 7, which are “Business” and “Management’s Discussion and Analysis of  Financial Condition and Results of Operations” respectively.


<p style="text-align: center">
    <img
    width=""
    height="300"
    src="https://i.imgur.com/ALyyjBZ.png">
    <div style="text-align: center"> Positive sentiment word cloud </div>

</p>

For positive sentiments, words like “increase”, “new”, and “growth” occur more frequently in positive sentiment. These words usually indicate advantageous information for the company.


<p style="text-align: center">
    <img
    width=""
    height="300"
    src="https://i.imgur.com/uUhxPK7.png">
    <div style="text-align: center"> Negative sentiment word cloud </div>

</p>


Whereas the word indicates uncertainty like “may”,  or negative meaning like “adversely”, or “risk” occur more frequently in negative sentiment. Moreover, common words “market”, “oper”- root for operation, and “rate” indicate that our model successfully captures the context of the sentences not guessing based on words only.



###### tags: `Templates` `Documentation`
