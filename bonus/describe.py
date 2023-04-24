import pandas as pd 
import datetime 


# Fun utils 
def calculate_age(born):
    today = datetime.date.today()
    age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    return age

 # Add variance 

 # Add Median 
 
def count(df, column): 
    return len(df[column].tolist())

# For Bonus 
def unique(df, column): 
    unique_values = set(df[column])
    num_unique_values = len(unique_values)
    return num_unique_values

# For Bonus 
def count_occurrences(df, column):
    occurrences = {}
    for value in df[column]:
        if value in occurrences:
            occurrences[value] += 1
        else:
            occurrences[value] = 1
    return occurrences

def mean_(df, column): 
    column_list = df[column].tolist() 
    mean = 0 
    for i in range(len(column_list)): 
        mean += column_list[i]
    mean = mean / len(column_list)
    return mean 

def std_dev(df, column): 
    column_list = df[column].tolist()
    mean = mean_(df, column) 
    std_dev = 0 
    for i in range(len(column_list)): 
        std_dev += (column_list[i] - mean) ** 2 
    std_dev = (std_dev / (len(column_list) - 1)) ** 0.5 
    return std_dev

def min(df, column): 
    column_list = df[column].tolist() 
    min = column_list[0] 
    for i in range(len(column_list)): 
        if column_list[i] < min: 
            min = column_list[i] 
    return min

def max(df, column): 
    column_list = df[column].tolist() 
    max = column_list[0] 
    for i in range(len(column_list)): 
        if column_list[i] > max: 
            max = column_list[i] 
    return max 

def median(df, column) : 
    x = df[column].tolist()
    x.sort() 
    if len(x) % 2 == 0:
        med1 = x[len(x)//2]
        med2 = x[len(x)//2 - 1]
        return (med1 + med2)/2 
    else :
        return x[len(x)//2]

def var(df, column) : 
    x = df[column].tolist()
    mean = sum(x)/len(x)  
    total = 0.0
    for value in x: 
        total = total + (value - mean)**2 
    return total/len(x) 


def percentile(df, column, percentile) : 
    column_list = df[column].tolist() 
    column_list.sort() 
    percentile = column_list[int(len(column_list) * percentile)] 
    return percentile 






if __name__ == "__main__" : 

    
    df = pd.read_csv( "dataset_train.csv" ) 
    df = df.dropna(axis=0, how='any')
    df = df.drop( [ "Hogwarts House" , "First Name" , "Last Name" , "Best Hand" ] , axis = 1 )  
    df['age'] = df['Birthday'].apply(lambda x: calculate_age(datetime.datetime.strptime(x, '%Y-%m-%d')))
    df = df.drop( [ "Birthday" ] , axis = 1 ) 
    df = df.drop( [ "Index" ] , axis = 1)

    data = {} 
   # in a for loop make a dictionnary with the column name as key and the function as value. 
    for column in df.columns: 
       data[column] = [count(df, column), mean_(df, column), median(df, column), std_dev(df, column),var(df, column), min(df, column), percentile(df, column, 0.25), percentile(df, column, 0.5), percentile(df, column, 0.75), max(df, column), unique(df, column) ]
    # create a dataframe with the dictionnary 
    df = pd.DataFrame(data, index = ['count', 'mean', 'median', 'std', 'var','min', '25%', '50%', '75%', 'max', 'unique']) 
    # print the dataframe 
    print(df) 





