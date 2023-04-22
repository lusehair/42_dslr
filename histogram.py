import pandas as pd 
import matplotlib.pyplot as plt 


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

    # Make a histogram who responds to the following: Which Hogwarts course has a homogeneous score distribution between all four houses? 

    df = pd.read_csv( "dataset_train.csv" ) 
    df = df.dropna(axis=0, how='any') 
    df = df.drop( ["First Name" , "Last Name" , "Best Hand", "Birthday"] , axis = 1 ) 
    # for each house, get the mean of each course. 

    # make a dataframe with the mean of each course for each house. 

    data = {} 

    # Do a z_score for each course.

    # for course in df.columns: 
    #     df[course] = z_score(df, course)


    houses = [ "Gryffindor" , "Slytherin" , "Ravenclaw" , "Hufflepuff" ] 
    courses = [ "Arithmancy" , "Astronomy" , "Herbology" , "Defense Against the Dark Arts" , "Divination" , "Muggle Studies" , "Ancient Runes" , "History of Magic" , "Transfiguration" , "Potions" , "Care of Magical Creatures" , "Charms" , "Flying" ] 
    for house in houses: 
        data[house] = [] 
        for course in courses: 
            # print(course, house, mean_(df[df[ "Hogwarts House" ] == house], course))
            data[house].append(mean_(df[df[ "Hogwarts House" ] == house], course)) 
    df = pd.DataFrame(data, index=courses) 
    df.plot.bar() 
    plt.show() 
    plt.savefig( "histogram.png" ) 



