# Manual Error Analysis

This report is generated from actual model predictions. Review and edit the qualitative observations before publishing.

## BERT truncated 512

### validation-1910.12574-81a35b9572c9d574a30cc2164f47750716157fc8

- **Question:** What existing approaches do they compare to?
- **Reference answers:** ["Waseem and Hovy BIBREF5, Davidson et al. BIBREF9, and Waseem et al. BIBREF10"]
- **Predicted answer:** existing approaches
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.417
- **Confidence proxy:** 0.033932
- **Answer-position bucket:** 4097+
- **Error category:** long-context-failure
- **Predicted evidence:** Generated hateful and toxic content by a portion of users in social media is a rising phenomenon that motivated researchers to dedicate substantial efforts to the challenging direction of hateful content identification. We not only need an efficient automatic hate speech detection model based on advanced machine learning and natural language processing, but also a sufficiently large amount of annotated data to train a model. The lack of a sufficient amount of labelled hate speech data, along with the existing biases, has been the main issue in this domain of research. To address these needs, in this study we introduce a novel transfer learning approach based on an existing pre-trained language model called BERT (Bidirectional Encoder Representations from Transformers). More specifically, we investigate the ability of BERT at capturing hateful context within social media content by using new fine-tuning methods based on transfer learning. To evaluate our proposed approach, we use two publicly available datasets that have been annotated for racism, sexism, hate, or offensive content on Twitter. The results show that our solution obtains considerable performance on these datasets in terms of precision and recall in comparison to existing approaches. Consequently, our model can capture some biases in data annotation and collection process and can potentially lead us to a more accurate model.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1902.09666-521280a87c43fcdf9f577da235e7072a23f0673e

- **Question:** How many annotators participated?
- **Reference answers:** ["five annotators"]
- **Predicted answer:** Offensive Language Identification Dataset (OLID)
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.244
- **Confidence proxy:** 0.035930
- **Answer-position bucket:** 2049-4096
- **Error category:** long-context-failure
- **Predicted evidence:** As offensive content has become pervasive in social media, there has been much research in identifying potentially offensive messages. However, previous work on this topic did not consider the problem as a whole, but rather focused on detecting very specific types of offensive content, e.g., hate speech, cyberbulling, or cyber-aggression. In contrast, here we target several different kinds of offensive content. In particular, we model the task hierarchically, identifying the type and the target of offensive messages in social media. For this purpose, we complied the Offensive Language Identification Dataset (OLID), a new dataset with tweets annotated for offensive content using a fine-grained three-layer annotation scheme, which we make publicly available. We discuss the main similarities and differences between OLID and pre-existing datasets for hate speech identification, aggression detection, and similar tasks. We further experiment with and we compare the performance of different machine learning models on OLID.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1910.08293-808f0ad46ca4eb4ea5492f9e14ca043fe1e206cc

- **Question:** How many different characters were in dataset?
- **Reference answers:** ["45,821 characters"]
- **Predicted answer:** three-component system, ALOHA
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.368
- **Confidence proxy:** 0.045501
- **Answer-position bucket:** 1025-2048
- **Error category:** long-context-failure
- **Predicted evidence:** For conversational AI and virtual assistants to communicate with humans in a realistic way, they must exhibit human characteristics such as expression of emotion and personality. Current attempts toward constructing human-like dialogue agents have presented significant difficulties. We propose Human Level Attributes (HLAs) based on tropes as the basis of a method for learning dialogue agents that can imitate the personalities of fictional characters. Tropes are characteristics of fictional personalities that are observed recurrently and determined by viewers' impressions. By combining detailed HLA data with dialogue data for specific characters, we present a dataset that models character profiles and gives dialogue agents the ability to learn characters' language styles through their HLAs. We then introduce a three-component system, ALOHA (which stands for Artificial Learning On Human Attributes), that combines character space mapping, character community detection, and language style retrieval to build a character (or personality) specific language model. Our preliminary experiments demonstrate that ALOHA, combined with our proposed dataset, can outperform baseline models at identifying correct dialogue responses of any chosen target character, and is stable regardless of the character's identity, genre of the show, and context of the dialogue.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1610.09225-4c07c33dfaf4f3e6db55e377da6fa69825d0ba15

- **Question:** What is the dimension of the embeddings?
- **Reference answers:** ["300"]
- **Predicted answer:** Dow Jones Industrial Average Index
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.263
- **Confidence proxy:** 0.061405
- **Answer-position bucket:** 2049-4096
- **Error category:** long-context-failure
- **Predicted evidence:** Predicting stock market movements is a well-known problem of interest. Now-a-days social media is perfectly representing the public sentiment and opinion about current events. Especially, twitter has attracted a lot of attention from researchers for studying the public sentiments. Stock market prediction on the basis of public sentiments expressed on twitter has been an intriguing field of research. Previous studies have concluded that the aggregate public mood collected from twitter may well be correlated with Dow Jones Industrial Average Index (DJIA). The thesis of this work is to observe how well the changes in stock prices of a company, the rises and falls, are correlated with the public opinions being expressed in tweets about that company. Understanding author's opinion from a piece of text is the objective of sentiment analysis. The present paper have employed two different textual representations, Word2vec and N-gram, for analyzing the public sentiments in tweets. In this paper, we have applied sentiment analysis and supervised machine learning principles to the tweets extracted from twitter and analyze the correlation between stock market movements of a company and sentiments in tweets. In an elaborate way, positive news and tweets in social media about a company would definitely encourage people to invest in the stocks of that company and as a result the stock price of that company would increase. At the end of the paper, it is shown that a strong correlation exists between the rise and falls in stock prices with the public sentiments in tweets.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1809.06537-06cc8fcafc0880cf69a2514bb7341642b9833041

- **Question:** what is the size of the real-world civil case dataset?
- **Reference answers:** ["INLINEFORM1 cases"]
- **Predicted answer:** In experiments
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.255
- **Confidence proxy:** 0.062102
- **Answer-position bucket:** 2049-4096
- **Error category:** long-context-failure
- **Predicted evidence:** Automatic judgment prediction aims to predict the judicial results based on case materials. It has been studied for several decades mainly by lawyers and judges, considered as a novel and prospective application of artificial intelligence techniques in the legal field. Most existing methods follow the text classification framework, which fails to model the complex interactions among complementary case materials. To address this issue, we formalize the task as Legal Reading Comprehension according to the legal scenario. Following the working protocol of human judges, LRC predicts the final judgment results based on three types of information, including fact description, plaintiffs' pleas, and law articles. Moreover, we propose a novel LRC model, AutoJudge, which captures the complex semantic interactions among facts, pleas, and laws. In experiments, we construct a real-world civil case dataset for LRC. Experimental results on this dataset demonstrate that our model achieves significant improvement over state-of-the-art models. We will publish all source codes and datasets of this work on \urlgithub.com for further research.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1806.05513-15cdd9ea4bae8891c1652da2ed34c87bbbd0edb8

- **Question:** Where did the texts in the corpus come from?
- **Reference answers:** ["tweets from the past two years from domains like `sports', `politics', `entertainment'", "twitter"]
- **Predicted answer:** English-Hindi
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.077
- **Confidence proxy:** 0.072173
- **Answer-position bucket:** 1025-2048
- **Error category:** long-context-failure
- **Predicted evidence:** Humor Detection in English-Hindi Code-Mixed Social Media Content : Corpus and Baseline System
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1604.05372-aefa333b2cf0a4000cd40566149816f5b36135e7

- **Question:** What evaluation metric do they use?
- **Reference answers:** ["ratio of correct `translations'"]
- **Predicted answer:** neural distributional models
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.300
- **Confidence proxy:** 0.079948
- **Answer-position bucket:** 2049-4096
- **Error category:** long-context-failure
- **Predicted evidence:** We present our experience in applying distributional semantics (neural word embeddings) to the problem of representing and clustering documents in a bilingual comparable corpus. Our data is a collection of Russian and Ukrainian academic texts, for which topics are their academic fields. In order to build language-independent semantic representations of these documents, we train neural distributional models on monolingual corpora and learn the optimal linear transformation of vectors from one language to another. The resulting vectors are then used to produce `semantic fingerprints' of documents, serving as input to a clustering algorithm. The presented method is compared to several baselines including `orthographic translation' with Levenshtein edit distance and outperforms them by a large margin. We also show that language-independent `semantic fingerprints' are superior to multi-lingual clustering algorithms proposed in the previous work, at the same time requiring less linguistic resources.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1701.05574-bbb77f2d6685c9257763ca38afaaef29044b4018

- **Question:** What is the best reported system?
- **Reference answers:** ["the MILR classifier"]
- **Predicted answer:** sarcasm detection by 3.7% (in terms of F-score), over the performance of the best reported system.
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.346
- **Confidence proxy:** 0.089649
- **Answer-position bucket:** 2049-4096
- **Error category:** long-context-failure
- **Predicted evidence:** In this paper, we propose a novel mechanism for enriching the feature vector, for the task of sarcasm detection, with cognitive features extracted from eye-movement patterns of human readers. Sarcasm detection has been a challenging research problem, and its importance for NLP applications such as review summarization, dialog systems and sentiment analysis is well recognized. Sarcasm can often be traced to incongruity that becomes apparent as the full sentence unfolds. This presence of incongruity- implicit or explicit- affects the way readers eyes move through the text. We observe the difference in the behaviour of the eye, while reading sarcastic and non sarcastic sentences. Motivated by his observation, we augment traditional linguistic and stylistic features for sarcasm detection with the cognitive features obtained from readers eye movement data. We perform statistical classification using the enhanced feature set so obtained. The augmented cognitive features improve sarcasm detection by 3.7% (in terms of F-score), over the performance of the best reported system.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1906.00790-0f567251a6566f65170a1329eeeb5105932036b2

- **Question:** What current state of the art method was used for comparison?
- **Reference answers:** ["current state-of-the-art approach BIBREF14 , BIBREF15"]
- **Predicted answer:** hashtag segmentation by framing it as a pairwise ranking problem between candidate segmentations. Our novel neural approaches demonstrate 24.6% error reduction in hashtag segmentation accuracy
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.310
- **Confidence proxy:** 0.090542
- **Answer-position bucket:** 513-1024
- **Error category:** long-context-failure
- **Predicted evidence:** Hashtags are often employed on social media and beyond to add metadata to a textual utterance with the goal of increasing discoverability, aiding search, or providing additional semantics. However, the semantic content of hashtags is not straightforward to infer as these represent ad-hoc conventions which frequently include multiple words joined together and can include abbreviations and unorthodox spellings. We build a dataset of 12,594 hashtags split into individual segments and propose a set of approaches for hashtag segmentation by framing it as a pairwise ranking problem between candidate segmentations. Our novel neural approaches demonstrate 24.6% error reduction in hashtag segmentation accuracy compared to the current state-of-the-art method. Finally, we demonstrate that a deeper understanding of hashtag semantics obtained through segmentation is useful for downstream applications such as sentiment analysis, for which we achieved a 2.6% increase in average recall on the SemEval 2017 sentiment analysis dataset.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1603.08594-9cf070d6671ee4a6353f79a165aa648309e01295

- **Question:** What is the size of the parallel corpus used to train the model constraints?
- **Reference answers:** ["1500 sentences"]
- **Predicted answer:** alignments from parallel data in another language
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.328
- **Confidence proxy:** 0.094617
- **Answer-position bucket:** 2049-4096
- **Error category:** long-context-failure
- **Predicted evidence:** In this paper, we attempt to solve the problem of Prepositional Phrase (PP) attachments in English. The motivation for the work comes from NLP applications like Machine Translation, for which, getting the correct attachment of prepositions is very crucial. The idea is to correct the PP-attachments for a sentence with the help of alignments from parallel data in another language. The novelty of our work lies in the formulation of the problem into a dual decomposition based algorithm that enforces agreement between the parse trees from two languages as a constraint. Experiments were performed on the English-Hindi language pair and the performance improved by 10% over the baseline, where the baseline is the attachment predicted by the MSTParser model trained for English.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1804.05918-f17ca24b135f9fe6bb25dc5084b13e1637ec7744

- **Question:** What discourse relations does it work best/worst for?
- **Reference answers:** ["explicit discourse relations"]
- **Predicted answer:** semantic meanings of a sentence or clause can not be interpreted independently from the rest of a paragraph
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.333
- **Confidence proxy:** 0.099298
- **Answer-position bucket:** within-first-512
- **Error category:** incorrect-answer
- **Predicted evidence:** We argue that semantic meanings of a sentence or clause can not be interpreted independently from the rest of a paragraph, or independently from all discourse relations and the overall paragraph-level discourse structure. With the goal of improving implicit discourse relation classification, we introduce a paragraph-level neural networks that model inter-dependencies between discourse units as well as discourse relation continuity and patterns, and predict a sequence of discourse relations in a paragraph. Experimental results show that our model outperforms the previous state-of-the-art systems on the benchmark corpus of PDTB.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1708.06185-e48e750743aef36529fbea4328b8253dbe928b4d

- **Question:** what dataset was used?
- **Reference answers:** ["WASSA-2017 Shared Task on Emotion Intensity"]
- **Predicted answer:** Twitter
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.300
- **Confidence proxy:** 0.103353
- **Answer-position bucket:** within-first-512
- **Error category:** incorrect-answer
- **Predicted evidence:** Twitter, a micro-blogging and social networking site has emerged as a platform where people express themselves and react to events in real-time. It is estimated that nearly 500 million tweets are sent per day . Twitter data is particularly interesting because of its peculiar nature where people convey messages in short sentences using hashtags, emoticons, emojis etc. In addition, each tweet has meta data like location and language used by the sender. It's challenging to analyze this data because the tweets might not be grammatically correct and the users tend to use informal and slang words all the time. Hence, this poses an interesting problem for NLP researchers. Any advances in using this abundant and diverse data can help understand and analyze information about a person, an event, a product, an organization or a country as a whole. Many notable use cases of the twitter can be found here.
- **Human review note:** _Add a concise explanation of the failure or success._

## Longformer SQuAD sliding windows

### validation-1801.07887-f67b9bda14ec70feba2e0d10c400b2b2025a0a6a

- **Question:** What downstream tasks are evaluated?
- **Reference answers:** ["text classification"]
- **Predicted answer:** Table~ SECREF4
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.286
- **Confidence proxy:** 0.050279
- **Answer-position bucket:** within-first-512
- **Error category:** incorrect-answer
- **Predicted evidence:** We ran BV2009 with smaller window sizes for each of our different batch sizes. Our results are summarized for a window size of one in the row ``BV2009 (Window Size = 1)'' in Table~ SECREF4 . When using a window size of one, BV2009 is able to stop with a smaller number of annotations than when using a window size of three. This is done without losing much F-Measure. The next subsection provides an explanation as to why smaller window sizes are more effective than larger window sizes when larger batch sizes are used.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1910.06592-d3ff2986ca8cb85a9a5cec039c266df756947b43

- **Question:** Based on this paper, what is the more predictive set of features to detect fake news?
- **Reference answers:** ["words embeddings, style, and morality features"]
- **Predicted answer:** probability distribution over the account types
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.286
- **Confidence proxy:** 0.072802
- **Answer-position bucket:** 2049-4096
- **Error category:** long-context-failure
- **Predicted evidence:** Given a news Twitter account, we read its tweets from the account's timeline. Then we sort the tweets by the posting date in ascending way and we split them into $N$ chunks. Each chunk consists of a sorted sequence of tweets labeled by the label of its corresponding account. We extract a set of features from each chunk and we feed them into a recurrent neural network to model the sequential flow of the chunks' tweets. We use an attention layer with dropout to attend over the most important tweets in each chunk. Finally, the representation is fed into a softmax layer to produce a probability distribution over the account types and thus predict the factuality of the accounts. Since we have many chunks for each account, the label for an account is obtained by taking the majority class of the account's chunks.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1610.09225-4c07c33dfaf4f3e6db55e377da6fa69825d0ba15

- **Question:** What is the dimension of the embeddings?
- **Reference answers:** ["300"]
- **Predicted answer:** Dow Jones Industrial Index
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.316
- **Confidence proxy:** 0.096668
- **Answer-position bucket:** 2049-4096
- **Error category:** long-context-failure
- **Predicted evidence:** The most well-known publication in this area is by Bollen BIBREF10 . They investigated whether the collective mood states of public (Happy, calm, Anxiety) derived from twitter feeds are correlated to the value of the Dow Jones Industrial Index. They used a Fuzzy neural network for their prediction. Their results show that public mood states in twitter are strongly correlated with Dow Jones Industrial Index. Chen and Lazer BIBREF11 derived investment strategies by observing and classifying the twitter feeds. Bing et al. BIBREF12 studied the tweets and concluded the predictability of stock prices based on the type of industry like Finance, IT etc. Zhang BIBREF13 found out a high negative correlation between mood states like hope, fear and worry in tweets with the Dow Jones Average Index. Recently, Brian et al. BIBREF14 investigated the correlation of sentiments of public with stock increase and decreases using Pearson correlation coefficient for stocks. In this paper, we took a novel approach of predicting rise and fall in stock prices based on the sentiments extracted from twitter to find the correlation. The core contribution of our work is the development of a sentiment analyzer which works better than the one in Brian's work and a novel approach to find the correlation. Sentiment analyzer is used to classify the sentiments in tweets extracted.The human annotated dataset in our work is also exhaustive. We have shown that a strong correlation exists between twitter sentiments and the next day stock prices in the results section. We did so by considering the tweets and stock opening and closing prices of Microsoft over a year.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-2002.01320-f8f4e4a50d2b3fbd193327e79ea32d8d057e1414

- **Question:** How was the dataset collected?
- **Reference answers:** ["Contributors record voice clips by reading from a bank of donated sentences.", "crowdsourcing"]
- **Predicted answer:** We extract 80-channel log-mel filterbank features, computed with a 25ms window size and 10ms window shift using torchaudio
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.263
- **Confidence proxy:** 0.196730
- **Answer-position bucket:** 513-1024
- **Error category:** long-context-failure
- **Predicted evidence:** We convert raw MP3 audio files from CoVo and TT into mono-channel waveforms, and downsample them to 16,000 Hz. For transcripts and translations, we normalize the punctuation, we tokenize the text with sacreMoses and lowercase it. For transcripts, we further remove all punctuation markers except for apostrophes. We use character vocabularies on all the tasks, with 100% coverage of all the characters. Preliminary experimentation showed that character vocabularies provided more stable training than BPE. For MT, the vocabulary is created jointly on both transcripts and translations. We extract 80-channel log-mel filterbank features, computed with a 25ms window size and 10ms window shift using torchaudio. The features are normalized to 0 mean and 1.0 standard deviation. We remove samples having more than 3,000 frames or more than 256 characters for GPU memory efficiency (less than 25 samples are removed for all languages).
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1611.02988-de53af4eddbc30c808d90b8a11a29217d377569e

- **Question:** Which Facebook pages did they look at?
- **Reference answers:** ["FoxNews, CNN, ESPN, New York Times, Time magazine, Huffington Post Weird News, The Guardian, Cartoon Network, Cooking Light, Home Cooking Adventure, Justin Bieber, Nickelodeon, Spongebob, Disney", "FoxNews, CNN, ESPN, New York Times, Time magazine, Huffington Post Weird News, The Guardian, Cartoon Network, Cooking Light, Home Cooking Adventure, Justin Bieber, Nickelodeon, Spongebob, Disney."]
- **Predicted answer:** Affective Text dataset, the Fairy Tales dataset
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.214
- **Confidence proxy:** 0.200399
- **Answer-position bucket:** 1025-2048
- **Error category:** long-context-failure
- **Predicted evidence:** Three datasets annotated with emotions are commonly used for the development and evaluation of emotion detection systems, namely the Affective Text dataset, the Fairy Tales dataset, and the ISEAR dataset. In order to compare our performance to state-of-the-art results, we have used them as well. In this Section, in addition to a description of each dataset, we provide an overview of the emotions used, their distribution, and how we mapped them to those we obtained from Facebook posts in Section SECREF7 . A summary is provided in Table TABREF8 , which also shows, in the bottom row, what role each dataset has in our experiments: apart from the development portion of the Affective Text, which we used to develop our models (Section SECREF4 ), all three have been used as benchmarks for our evaluation.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1902.09666-5a8cc8f80509ea77d8213ed28c5ead501c68c725

- **Question:** What is the definition of offensive language?
- **Reference answers:** ["Most prior work focuses on a different aspect of offensive language such as abusive language BIBREF0 , BIBREF1 , (cyber-)aggression BIBREF2 , (cyber-)bullying BIBREF3 , BIBREF4 , toxic comments INLINEFORM0 , hate speech BIBREF5 , BIBREF6 , BIBREF7 , BIBREF8 , BIBREF9 , BIBREF10 , and offensive language BIBREF11 . Prior work has focused on these aspects of offensive language in Twitter BIBREF3 , BIBREF7 , BIBREF8 , BIBREF11 , Wikipedia comments, and Facebook posts BIBREF2 ."]
- **Predicted answer:** Target Identification
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.048
- **Confidence proxy:** 0.207676
- **Answer-position bucket:** within-first-512
- **Error category:** incorrect-answer
- **Predicted evidence:** Offensive Language Target Identification
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1908.08717-f85f2a532e7e700d9f8f9c09cd08d4e47b87bdd3

- **Question:** What is the goal of investigating NLP gender bias specifically in the news broadcast domain and Anchor role?
- **Reference answers:** ["create fair systems"]
- **Predicted answer:** word embeddings BIBREF5 and semantics BIBREF6
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.318
- **Confidence proxy:** 0.208774
- **Answer-position bucket:** 4097+
- **Error category:** long-context-failure
- **Predicted evidence:** The gender issue has returned to the forefront of the media scene in recent years and with the emergence of AI technologies in our daily lives, gender bias has become a scientific topic that researchers are just beginning to address. Several studies revealed the existence of gender bias in AI technologies such as face recognition (GenderShades BIBREF17), NLP (word embeddings BIBREF5 and semantics BIBREF6) and machine translation (BIBREF18, BIBREF7). The impact of the training data used within these deep-learning algorithms is therefore questioned.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1604.03114-26327ccebc620a73ba37a95aabe968864e3392b2

- **Question:** what aspects of conversation flow do they look at?
- **Reference answers:** ["—promoting one's own points and attacking the opponents' points"]
- **Predicted answer:** words that become relevant to the conversation
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.250
- **Confidence proxy:** 0.211292
- **Answer-position bucket:** within-first-512
- **Error category:** incorrect-answer
- **Predicted evidence:** Identifying discussion points. Having seen that debaters can benefit by shifting away from talking points that were salient during the introduction, we now examine the ideas that spontaneously arise to serve the discussion. We model such discussion points as words introduced to the debate during the discussion by a debater and adopted by his opponents at least twice. This allows us to focus on words that become relevant to the conversation; only 3% of all newly introduced words qualify, amounting to about 10 discussion points per debate.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1707.06806-ddb23a71113cbc092cbc158066d891cae261e2c6

- **Question:** What is the source of the news articles?
- **Reference answers:** ["main news channels, such as Yahoo News, The Guardian or The Washington Post", "The BreakingNews dataset"]
- **Predicted answer:** title
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.417
- **Confidence proxy:** 0.229987
- **Answer-position bucket:** 2049-4096
- **Error category:** long-context-failure
- **Predicted evidence:** With the ever decreasing attention span of contemporary Internet users, the title of online content (such as a news article or video) can be a major factor in determining its popularity. To take advantage of this phenomenon, we propose a new method based on a bidirectional Long Short-Term Memory (LSTM) neural network designed to predict the popularity of online content using only its title. We evaluate the proposed architecture on two distinct datasets of news articles and news videos distributed in social media that contain over 40,000 samples in total. On those datasets, our approach improves the performance over traditional shallow approaches by a margin of 15%. Additionally, we show that using pre-trained word vectors in the embedding layer improves the results of LSTM models, especially when the training set is small. To our knowledge, this is the first attempt of applying popularity prediction using only textual information from the title.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1702.03856-8acab64ba72831633e8cc174d5469afecccf3ae9

- **Question:** what is the domain of the corpus?
- **Reference answers:** ["telephone calls"]
- **Predicted answer:** the pseudotext does not represent all of the source words
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.372
- **Confidence proxy:** 0.245667
- **Answer-position bucket:** 513-1024
- **Error category:** long-context-failure
- **Predicted evidence:** In creating a translation model from this data, we face a difficulty that does not arise in the parallel texts that are normally used to train translation models: the pseudotext does not represent all of the source words, since the discovered segments do not cover the full audio (Fig. FIGREF4 ). Hence we must not assume that our MT model can completely recover the translation of a test sentence. In these conditions, the language modeling and ordering assumptions of most MT models are unwarranted, so we instead use a simple bag-of-words translation model based only on co-occurrence: IBM Model 1 BIBREF11 with a Dirichlet prior over translation distributions, as learned by fast_align BIBREF12 . In particular, for each pseudoterm, we learn a translation distribution over possible target words. To translate a pseudoterm in test data, we simply return its highest-probability translation (or translations, as discussed in § SECREF5 ).
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1909.00105-c77359fb9d3ef96965a9af0396b101f82a0a9de6

- **Question:** How did they obtain the interactions?
- **Reference answers:** ["from Food.com"]
- **Predicted answer:** users were given a partial recipe specification
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.368
- **Confidence proxy:** 0.254629
- **Answer-position bucket:** 2049-4096
- **Error category:** long-context-failure
- **Predicted evidence:** We prepared a set of 15 pairwise comparisons per evaluation session, and collected 930 pairwise evaluations (310 per personalized model) over 62 sessions. For each pair, users were given a partial recipe specification (name and 3-5 key ingredients), as well as two generated recipes labeled `A' and `B'. One recipe is generated from our baseline encoder-decoder model and one recipe is generated by one of our three personalized models (Prior Tech, Prior Name, Prior Recipe). The order of recipe presentation (A/B) is randomly selected for each question. A screenshot of the user evaluation interface is given in fig:exeval. We ask the user to indicate which recipe they find more coherent, and which recipe best accomplishes the goal indicated by the recipe name. A screenshot of this survey interface is given in fig:exeval2.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1911.06964-2ad4d3d222f5237ed97923640bc8e199409cbe52

- **Question:** What user variations have been tested?
- **Reference answers:** ["completion times and accuracies"]
- **Predicted answer:** is robust
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.273
- **Confidence proxy:** 0.257469
- **Answer-position bucket:** 2049-4096
- **Error category:** long-context-failure
- **Predicted evidence:** We study textual autocomplete---the task of predicting a full sentence from a partial sentence---as a human-machine communication game. Specifically, we consider three competing goals for effective communication: use as few tokens as possible (efficiency), transmit sentences faithfully (accuracy), and be learnable to humans (interpretability). We propose an unsupervised approach which tackles all three desiderata by constraining the communication scheme to keywords extracted from a source sentence for interpretability and optimizing the efficiency-accuracy tradeoff. Our experiments show that this approach results in an autocomplete system that is 52% more accurate at a given efficiency level compared to baselines, is robust to user variations, and saves time by nearly 50% compared to typing full sentences.
- **Human review note:** _Add a concise explanation of the failure or success._

## Longformer QASPER fine-tuned

### validation-1902.09666-55bd59076a49b19d3283af41c5e3ccb875f3eb0c

- **Question:** What is the best performing model?
- **Reference answers:** ["CNN"]
- **Predicted answer:** bidirectional Long Short-Term-Memory (BiLSTM) model
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.244
- **Confidence proxy:** 0.122007
- **Answer-position bucket:** 2049-4096
- **Error category:** long-context-failure
- **Predicted evidence:** We assess our dataset using traditional and deep learning methods. Our simplest model is a linear SVM trained on word unigrams. SVMs have produced state-of-the-art results for many text classification tasks BIBREF13 . We also train a bidirectional Long Short-Term-Memory (BiLSTM) model, which we adapted from the sentiment analysis system of sentimentSystem,rasooli2018cross and altered to predict offensive labels instead. It consists of (1) an input embedding layer, (2) a bidirectional LSTM layer, (3) an average pooling layer of input features. The concatenation of the LSTM's and average pool layer is passed through a dense layer and the output is passed through a softmax function. We set two input channels for the input embedding layers: pre-trained FastText embeddings BIBREF14 , as well as updatable embeddings learned by the model during training. Finally, we also apply a Convolutional Neural Network (CNN) model based on the architecture of BIBREF15 , using the same multi-channel inputs as the above BiLSTM.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1908.09246-15e481e668114e4afe0c78eefb716ffe1646b494

- **Question:** What alternative to Gibbs sampling is used?
- **Reference answers:** ["generator network to capture the event-related patterns"]
- **Predicted answer:** INLINEFORM3 could be obtained by sampling uniformly along a straight line between INLINEFORM4 and INLINEFORM5
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.100
- **Confidence proxy:** 0.159633
- **Answer-position bucket:** 1025-2048
- **Error category:** long-context-failure
- **Predicted evidence:** where INLINEFORM0 denotes the discriminator loss, INLINEFORM1 represents the gradient penalty regularization loss, INLINEFORM2 is the gradient penalty coefficient which trade-off the two components of objective, INLINEFORM3 could be obtained by sampling uniformly along a straight line between INLINEFORM4 and INLINEFORM5 , INLINEFORM6 denotes the corresponding distribution.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1809.01541-32a3c248b928d4066ce00bbb0053534ee62596e7

- **Question:** What is MSD prediction?
- **Reference answers:** ["morphosyntactic descriptions (MSD)"]
- **Predicted answer:** the tag PRO;NOM;SG;1 is predicted as a sequence of four components, INLINEFORM0 PRO, NOM, SG, 1 INLINEFORM1
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.400
- **Confidence proxy:** 0.164365
- **Answer-position bucket:** within-first-512
- **Error category:** incorrect-answer
- **Predicted evidence:** We introduce an auxiliary objective that is meant to increase the morpho-syntactic awareness of the encoder and to regularise the learning process—the task is to predict the MSD tag of the target form. MSD tag predictions are conditioned on the context encoding, as described in UID15 . Tags are generated with an LSTM one component at a time, e.g. the tag PRO;NOM;SG;1 is predicted as a sequence of four components, INLINEFORM0 PRO, NOM, SG, 1 INLINEFORM1 .
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1910.05603-53f74250948015c394e7b8438a2041fdeb330911

- **Question:** What are the deep learning architectures used in the task?
- **Reference answers:** ["DNN-based acoustic model BIBREF0"]
- **Predicted answer:** Kaldi speech recognition toolkitBIBREF2
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.136
- **Confidence proxy:** 0.167054
- **Answer-position bucket:** within-first-512
- **Error category:** incorrect-answer
- **Predicted evidence:** There are two different testing sets from VLSP 2018 and VLSP 2019. In general, the data of this year is more complex than the last year one, so there is a big gap in results between two of them. The experiments are conducted using the Kaldi speech recognition toolkitBIBREF2.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1806.04511-c7486d039304ca9d50d0571236429f4f6fbcfcf7

- **Question:** which non-english language was the had the worst results?
- **Reference answers:** ["Turkish"]
- **Predicted answer:** Dutch
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.294
- **Confidence proxy:** 0.183816
- **Answer-position bucket:** within-first-512
- **Error category:** incorrect-answer
- **Predicted evidence:** We found that 25.84% of Dutch, 21.76% of Turkish, 24.46% Spanish, and 10.71% of Russian reviews that were misclassified had non-English words in them. These non-English words might be causing the misclassifications. However, a large portion of the missclassifications is not caused due to not-translated words. At the end, the machine translation errors has some but not noticeable effects on our model. Therefore, we can claim that machine translation preserves most of the information necessary for sentiment analysis.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1903.03530-dd53baf26dad3d74872f2d8956c9119a27269bd5

- **Question:** How do they select instances to their hold-out test set?
- **Reference answers:** ["held out from the simulated data"]
- **Predicted answer:** one inquiry template and one response template are randomly chosen to compose a turn
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.115
- **Confidence proxy:** 0.190242
- **Answer-position bucket:** 4097+
- **Error category:** long-context-failure
- **Predicted evidence:** (b) Template Selection: For each selected topic, one inquiry template and one response template are randomly chosen to compose a turn. To minimize adverse effects of underfitting, we redistributed the frequency distribution in Table TABREF14 : For utterance types that are below 15%, we boosted them to 15%, and the overall relative distribution ranking is balanced and consistent with Table TABREF14 .
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1909.05359-03ebb29c08375afc42a957c7b2dc1a42bed7b713

- **Question:** How is the effectiveness of this pipeline approach evaluated?
- **Reference answers:** ["proposed ontology, which, in our evaluation procedure, was populated with 3121 events entries from 51 documents."]
- **Predicted answer:** it receives sentences as input and outputs ontological information
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.105
- **Confidence proxy:** 0.190823
- **Answer-position bucket:** 2049-4096
- **Error category:** long-context-failure
- **Predicted evidence:** This framework, developed within the context of the Agatha project (described in Section SECREF1) has the full processing pipeline for Portuguese texts: it receives sentences as input and outputs ontological information: a) first performs all NLP typical tasks until semantic role labelling; b) then, it extracts subject-verb-object triples; c) and, then, it performs ontology matching procedures. As a final result, the obtained output is inserted into a specialized ontology.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-2003.11645-ef3567ce7301b28e34377e7b62c4ec9b496f00bf

- **Question:** What Named Entity Recognition dataset is used?
- **Reference answers:** ["Groningen Meaning Bank", "Groningen Meaning Bank (GMB)"]
- **Predicted answer:** Wiki Abstract dataset
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.221
- **Confidence proxy:** 0.192829
- **Answer-position bucket:** 513-1024
- **Error category:** long-context-failure
- **Predicted evidence:** To form the vocabulary, words occurring less than 5 times in the corpora were dropped, stop words removed using the natural language toolkit (NLTK) (BIBREF22) and data pre-processing carried out. Table TABREF2 describes most hyper-parameters explored for each dataset. In all, 80 runs (of about 160 minutes) were conducted for the 15MB Wiki Abstract dataset with 80 serialized models totaling 15.136GB while 80 runs (for over 320 hours) were conducted for the 711MB SW dataset, with 80 serialized models totaling over 145GB. Experiments for all combinations for 300 dimensions were conducted on the 3.9GB training set of the BW corpus and additional runs for other dimensions for the window 8 + skipgram + heirarchical softmax combination to verify the trend of quality of word vectors as dimensions are increased.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1908.11047-86bf75245358f17e35fc133e46a92439ac86d472

- **Question:** What are improvements for these two approaches relative to ELMo-only baselines?
- **Reference answers:** ["only modest gains on three of the four downstream tasks", "the performance differences across all tasks are small enough"]
- **Predicted answer:** transformers BIBREF16
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.140
- **Confidence proxy:** 0.203142
- **Answer-position bucket:** 513-1024
- **Error category:** long-context-failure
- **Predicted evidence:** Our model uses two encoders: $e_{\mathit {seq}}$ for encoding the sequential history ($x_{<i}$), and $e_{\mathit {syn}}$ for shallow syntactic (chunk) history ($c_{\leqslant i}$). For both, we use transformers BIBREF16, which consist of large feedforward networks equipped with multiheaded self-attention mechanisms.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1912.00667-67e9e147b2cab5ba43572ce8a17fc863690172f0

- **Question:** How are the interpretability merits of the approach demonstrated?
- **Reference answers:** ["directly solicits informative keywords from the crowd for model training, thereby providing human-understandable explanations for the improved model"]
- **Predicted answer:** Our approach improves LR by 5.17% (Accuracy) and 18.38% (AUC), and MLP by 10.71% (Accuracy) and 30.27% (AUC) on average
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.444
- **Confidence proxy:** 0.210292
- **Answer-position bucket:** 4097+
- **Error category:** long-context-failure
- **Predicted evidence:** Our approach improves LR by 5.17% (Accuracy) and 18.38% (AUC), and MLP by 10.71% (Accuracy) and 30.27% (AUC) on average. Such significant improvements clearly demonstrate that our approach is effective at improving model performance. We observe that the target models generally converge between the 7th and 9th iteration on both datasets when performance is measured by AUC. The performance can slightly degrade when the models are further trained for more iterations on both datasets. This is likely due to the fact that over time, the newly discovered keywords entail lower novel information for model training. For instance, for the CyberAttack dataset the new keyword in the 9th iteration `election' frequently co-occurs with the keyword `russia' in the 5th iteration (in microposts that connect Russian hackers with US elections), thus bringing limited new information for improving the model performance. As a side remark, we note that the models converge faster when performance is measured by accuracy. Such a comparison result confirms the difference between the metrics and shows the necessity for more keywords to discriminate event-related microposts from non event-related ones.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1609.00559-88e62ea7a4d1d2921624b8480b5c6b50cfa5ad42

- **Question:** What is a second order co-ocurrence matrix?
- **Reference answers:** ["frequencies of the other words which occur with both of them (i.e., second order co–occurrences)"]
- **Predicted answer:** INLINEFORM0 similarity matrix
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.134
- **Confidence proxy:** 0.212702
- **Answer-position bucket:** 513-1024
- **Error category:** long-context-failure
- **Predicted evidence:** We build an INLINEFORM0 similarity matrix using an external corpus where the rows and columns represent words within the corpus and the element contains the similarity score between the row word and column word using the similarity measures discussed above. If a word maps to more than one possible sense, we use the sense that returns the highest similarity score.
- **Human review note:** _Add a concise explanation of the failure or success._

### validation-1910.06036-1d197cbcac7b3f4015416f0152a6692e881ada6c

- **Question:** How they extract "structured answer-relevant relation"?
- **Reference answers:** ["off-the-shelf toolbox of OpenIE"]
- **Predicted answer:** extracted from the sentence in advance
- **Exact Match:** 0.000
- **Token F1:** 0.000
- **Evidence recovered:** 0.000
- **Evidence token recall:** 0.404
- **Confidence proxy:** 0.213498
- **Answer-position bucket:** 1025-2048
- **Error category:** long-context-failure
- **Predicted evidence:** Question generation (QG) is the task of generating a question from a reference sentence and a specified answer within the sentence. A major challenge in QG is to identify answer-relevant context words to finish the declarative-to-interrogative sentence transformation. Existing sequence-to-sequence neural models achieve this goal by proximity-based answer position encoding under the intuition that neighboring words of answers are of high possibility to be answer-relevant. However, such intuition may not apply to all cases especially for sentences with complex answer-relevant relations. Consequently, the performance of these models drops sharply when the relative distance between the answer fragment and other non-stop sentence words that also appear in the ground truth question increases. To address this issue, we propose a method to jointly model the unstructured sentence and the structured answer-relevant relation (extracted from the sentence in advance) for question generation. Specifically, the structured answer-relevant relation acts as the to the point context and it thus naturally helps keep the generated question to the point, while the unstructured sentence provides the full information. Extensive experiments show that to the point context helps our question generation model achieve significant improvements on several automatic evaluation metrics. Furthermore, our model is capable of generating diverse questions for a sentence which conveys multiple relations of its answer fragment.
- **Human review note:** _Add a concise explanation of the failure or success._
