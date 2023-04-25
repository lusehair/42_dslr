import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

if __name__ == "__main__" : 


    df = pd.read_csv( "dataset_train.csv" ) 
    df = df.dropna(axis=0, how='any') 
    df = df.drop( ["First Name" , "Last Name" , "Best Hand", "Birthday"] , axis = 1 ) 

    sns.pairplot(df, hue="Hogwarts House") 
    plt.show() 
    plt.savefig("pair_plot.png")

    # At the end wee keep Charms, Ancient Runes, herbology and Astronomy 
    print("We Keep Charms, Ancient Runes, herbology and Astronomy") 