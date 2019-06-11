# NBA-player-analysis

## Data
There are NBA player datas of 3 seasons. (16-17 season, 17-18 season, 18-19 season)
Winning rate of each team in 16-19 seasons.

data size:
- 16-17 season: 29x500
- 17-18 season: 29x540
- 18-19 season: 29x530

(source:https://stats.nba.com/)

## Method

### 1. Cluster the class of players (Unsupervised training)
First, we use 3 by 3 __miniSOM__ to train all of the player data. Found that there are 3 main perceptron were activate the most.

### 2. Train with __K-means__
Base on the result of SOM, we gave k=3 to k-means and classify all the players into 3 groups.

### 3. Analyse the team data
Count the amount of type of players for each team. Than give the winning rate to each team.

### 4. Supervised training
Use different regression to train the team data, and try to predict the team winning rate through the models.


