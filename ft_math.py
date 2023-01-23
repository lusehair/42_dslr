from datetime import datetime

def ft_mean(numbers) :
    return sum(numbers) / float(len(numbers)) 

def ft_sample_std_dev(numbers) :
    n = len(numbers)
    mean = ft_mean(numbers)
    return (sum((x - mean) ** 2 for x in numbers) / (n - 1)) ** 0.5  

def ft_quantile(x, p) :
    p_index = int(p * len(x))
    return sorted(x)[p_index] 

def ft_min(x) :
    return ft_quantile(x, 0) 

def ft_max(x) :
    return ft_quantile(x, 1) 

def date_to_age(date) :
    return (datetime.now() - date).days / 365.25 