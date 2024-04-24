# RAG-based-system-for-explainable-Movie-Search

## Hyperlinks
* [Meetings notes](https://docs.google.com/document/d/13XV529CzxseJ4wfdnEoQAKRNz8hCcIXpr1cMPGujlcE/edit?usp=sharing)
* [Timeline](https://docs.google.com/document/d/1aeqbfDCny26YenkR2OS5SquHwYyDr2x-pAUu2QmLaPs/edit)
* [Paper](https://www.overleaf.com/5328894874bzgyjztmzxbb#f507a1)
***

## Launch instructions
1. First clone the repository using the following command:
```sh
git clone https://github.com/Tsalyk/RAG-based-system-for-explainable-Movie-Search.git
```
2. Navigate clonned directory ```RAG-based-system-for-explainable-Movie-Search```
3. Launch [Docker](https://www.docker.com/)
4. Build docker containers with a command:
```sh
docker-compose up -d
```
in the root of navigated directory (approximately ~15 minutes)

5. Expose LLM server running [Google Colab notebook](https://colab.research.google.com/drive/1KZYaEtJDWsxzc9N3CWEIbcaVu2ipGgzG?usp=sharing) (select GPU in resources tab, approximately ~10 minutes)
6. Navigate http://localhost:8501/ and have fun testing out the application

```Note: steps 4 and 5 could be performed in parallel, you do not need to wait untill step 3 is finished```
***

## Overview
Usual key-words search engines can deliver results
based on popularity or broad relevancy, but they frequently can’t recognise and ac-
commodate personal preferences or query context, which leads to poor user expe-
riences. However, it is not enough just to provide semantically meaningful results
for the users, since they might not catch the reason why these movies are relevant.
By incorporating explainability into our movie search system, we aim to bridge the
gap between users’ search queries and the search results provided. Through trans-
parent reasoning for each movie suggestion, users gain valuable insights into the
decision-making process of the system. This not only fosters trust in the search en-
gine, but also empowers users to make more informed decisions about the movies
they choose to watch. Furthermore, explainability enables users to refine their search
criteria, leading to more personalized and tailored suggestions over time
***

## Diagram
<img src="https://github.com/Tsalyk/DomainSpecificAIAssistant/blob/main/assets/AIMovieSearch.png" width="700" height="600">
