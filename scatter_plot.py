import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 


def mean_(df, column): 
    column_list = df[column].tolist() 
    mean = 0 
    for i in range(len(column_list)): 
        mean += column_list[i]
    mean = mean / len(column_list)
    return mean 

def std_dev_(df, column): 
    column_list = df[column].tolist()
    mean = mean_(df, column) 
    std_dev = 0 
    for i in range(len(column_list)): 
        std_dev += (column_list[i] - mean) ** 2 
    std_dev = (std_dev / (len(column_list) - 1)) ** 0.5 
    return std_dev  

def z_score(df, column): 
    column_list = df[column].tolist() 
    mean = mean_(df, column) 
    std_dev = std_dev_(df, column) 
    z_score = [] 
    for i in range(len(column_list)): 
        z_score.append((column_list[i] - mean) / std_dev) 
    return z_score 

if __name__ == "__main__" : 


    df = pd.read_csv( "dataset_train.csv" ) 
    df = df.dropna(axis=0, how='any') 
    df = df.drop( ["First Name" , "Last Name" , "Best Hand", "Birthday"] , axis = 1 ) 


    # Make a scatter plot between two courses for each house. 

    # for each house, get the mean of each course. 
    # make a dataframe with the mean of each course for each house. 
    data = {}
    houses = [ "Gryffindor" , "Slytherin" , "Ravenclaw" , "Hufflepuff" ]
    courses = [ "Arithmancy" , "Astronomy" , "Herbology" , "Defense Against the Dark Arts" , "Divination" , "Muggle Studies" , "Ancient Runes" , "History of Magic" , "Transfiguration" , "Potions" , "Care of Magical Creatures" , "Charms" , "Flying" ]
    # normalize the data.
    for course in courses:
        df[course] = z_score(df, course)
    # make a scatter plot for each course for each house and index 
    for house in houses: 
        data[house] = [] 
        for course in courses:
            data[house].append(df[df[ "Hogwarts House" ] == house][course].tolist()) 
    # make a scatter plot for each house. 
    for house in houses: 
        for i in range(len(courses)): 
            for j in range(len(courses)): 
                if i != j: 
                    plt.scatter(data[house][i], data[house][j]) 
                    plt.title( "Scatter plot of {} and {} for {}" .format(courses[i], courses[j], house)) 
                    plt.xlabel(courses[i]) 
                    plt.ylabel(courses[j]) 
                    plt.show() 
                    plt.clf() 
                    plt.cla() 
                    plt.close() 

