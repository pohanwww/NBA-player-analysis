import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from sklearn import preprocessing
from sklearn.cluster import KMeans
from minisom import MiniSom
from pandas import DataFrame
import pickle
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble

k = 3
def preprocess_data():
    data_1617 = pd.read_csv('nba_player_1617.csv')
    data_1617.insert(0,'SEASON','1617')
    data_1718 = pd.read_csv('nba_player_1718.csv')
    data_1718.insert(0,'SEASON','1718')
    data_1819 = pd.read_csv('nba_player_1819.csv')
    data_1819.insert(0,'SEASON','1819')
    data = pd.concat([data_1617,data_1718,data_1819]) #1570 row
    data = data.drop(columns=['AGE','GP','W','L','MIN','FGM','FGA','3PM','3PA','3P%','FTM','FTA','FT%','OREB','DREB','STL','BLK','PF','FP','DD2','TD3','+/-'])
    data_list = data.values.tolist()
    
    return data_list
    
def player_analysis(data):
    for h in range(k):
        globals()['class_{}'.format(h)] = []
        globals()['count_{}'.format(h)] = 0
        for item in data:
            if item[-1] == h:
                globals()['class_{}'.format(h)].append(item)
                globals()['count_{}'.format(h)] += 1
    for h in range(k):
        print('class_{}:'.format(h),globals()['count_{}'.format(h)])
    avg = [] #avg data of each class
    for h in range(k):
        globals()['avg_{}'.format(h)] = []
        for i in range(3,len(data[0])):
            average = sum(x[i] for x in globals()['class_{}'.format(h)])/len(globals()['class_{}'.format(h)])
            globals()['avg_{}'.format(h)].append(average)
        avg.append(globals()['avg_{}'.format(h)])
    
    df = pd.DataFrame(avg)
    df.columns = ['PTS','FG%','REB','AST','TOV','k']
    df.index = ['class_1','class_2','class_3']
    df.to_csv("player_analysis.csv", index=True)    #output the player analysis results

def team_analysis(data):
    le = preprocessing.LabelEncoder()
    team_list = le.fit_transform([x[2] for x in data])

    for i in range(90): #30 teams each season, 3 seasons got 90 teams
        globals()['team_{}'.format(i)] = [0]*k
    for index, item in enumerate(data):
        for h in range(k):
            if item[8] == h:
                if item[0] == '1617':
                    globals()['team_{}'.format(team_list[index])][h] +=1
                elif item[0] == '1718':
                    globals()['team_{}'.format(team_list[index]+30)][h] +=1
                elif item[0] == '1819':
                    globals()['team_{}'.format(team_list[index]+60)][h] +=1     
            else:
                continue
    team = []
    team_winning_rate = pd.read_csv('team_winning_rate.csv')
    team_winRate_list = team_winning_rate.values.tolist()
    for i in range(90):
        globals()['team_{}'.format(i)].append(team_winRate_list[i][2])
        team.append(globals()['team_{}'.format(i)])

    df2 = pd.DataFrame(team)
    df2.columns = ['TYPE_A','TYPE_B','TYPE_C','winRate']
    df2.to_csv('team_analysis.csv',index=True)

    return df2

def try_different_method(model,x_train,x_test,y_train,y_test):
    model.fit(x_train,y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    print('model:',model)
    print('score:',score)
    print(result)
    plt.figure()
    plt.plot(np.arange(len(result)), y_test,'go-',label='true value')
    plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
    plt.title('score: %f'%score)
    plt.legend()
    plt.show()

def main():
    data_withTeam = preprocess_data()
    data = []
    for item in data_withTeam:
        data.append(item[3:])
    data_array = np.array(data)

    # #SOM
    # som = MiniSom(3 , 3, len(data_array[0]), sigma=0.3, learning_rate=0.2)
    # som.random_weights_init(data_array)
    # starting_weights = som.get_weights().copy()
    # # print('weight:',starting_weights)
    # som.train_random(data_array, len(data)*180, verbose=True)   #iteration=1570*180=282600
    # qnt = som.quantization(data_array)
    # # print('qnt:',qnt)
    # # map = som.distance_map(data)
    # result1 = som.activation_response(data_array)
    # print('result1 are: \n', result1)
    # print(som.get_weights())

    # #K Means
    # kmeans = KMeans(n_clusters=k, init = 'random', n_init = 1)
    # kmeans = kmeans.fit(data)
    # with open('kmeans.p', 'wb') as outfile:
    #     pickle.dump(kmeans, outfile)
    with open('kmeans.p', 'rb') as outfile:
        kmeans = pickle.load(outfile)

    #label the data
    labels = kmeans.predict(data)
    for i in range(len(labels)):
        data[i].append(labels[i])
        data_withTeam[i].append(labels[i])

    #plot data
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter([x[0] for x in data],[x[1] for x in data],[x[2] for x in data],c=labels, axis="PTS")
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter([x[1] for x in data],[x[2] for x in data],[x[3] for x in data],c=labels)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter([x[2] for x in data],[x[3] for x in data],[x[4] for x in data],c=labels)
    plt.show()

    #analyse data
    player_analysis(data_withTeam)
    df2 = team_analysis(data_withTeam)

    #Supervised learning
    data_list = df2.values.tolist()
    x_team = []
    y_team = []
    
    for i in range(len(data_list)):
        x_team.append(data_list[i][0:-1])
        y_team.append(data_list[i][-1])
    
    x_train,x_test,y_train,y_test = train_test_split(x_team, y_team, test_size = 0.3)

    #Create models
    model_LinearRegression = linear_model.LinearRegression()
    model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
    model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
    model_SVR = svm.SVR()
    model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20) 
    
    #train different models
    try_different_method(model_LinearRegression,x_train,x_test,y_train,y_test)
    try_different_method(model_KNeighborsRegressor,x_train,x_test,y_train,y_test)
    # try_different_method(model_DecisionTreeRegressor,x_train,x_test,y_train,y_test)
    # try_different_method(model_SVR,x_train,x_test,y_train,y_test)
    # try_different_method(model_RandomForestRegressor,x_train,x_test,y_train,y_test)

main()