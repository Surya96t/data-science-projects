import pandas as pd
import numpy as np
import scipy.stats as ss
import tensorflow as tf


def calculate_mean(arr):
    n = len(arr)
    # check for empty array 
    if n == 0:
        return None
    
    total = 0
    for num in arr:
        total += num
        
        
    mean = total/n
    
    return mean

def calculate_median(arr):
    n = len(arr)
    # check for empty array 
    if n == 0:
        return None
    
    # sort the array 
    sorted_arr = sorted(arr)
    
    # If list is odd return middle val
    # else return average of two mid vals
    # // -> Floor division Ex: 6.4 -> 6
    if n%2 == 1:
        return sorted_arr[n // 2]
    else:
        mid1 = sorted_arr[(n // 2)-1]
        mid2 = sorted_arr[(n // 2)]
        
        return (mid1 + mid2)/2

    
def calculate_var(arr):
    n = len(arr)
    # variance requires atleast 2 data points
    # check for required data points
    if n < 2:
        return None
    
    mean = calculate_mean(arr)
    variance = sum((x - mean) ** 2 for x in arr) / n
   
    return variance 

def calculate_std_dev(arr):
    n = len(arr)
    # std_dev = (var) ** 0.5
    if n < 2:
        return None
    
    var = calculate_var(arr)
    
    return (var)**0.5

def pearson_coeff_r(x_arr, y_arr):
    # check both arr are of same length
    if len(x_arr) != len(y_arr) or len(x_arr) < 2:
        raise Exception("Length of two arrays does not match")
    
    n = len(x_arr)
    
    
    sum_x = sum(x_arr)
    sum_y = sum(y_arr)
    sum_xy = sum(x * y for x, y in zip(x_arr, y_arr))
    
    numerator = n * sum_xy - sum_x * sum_y
    
    sum_x_sq = sum(x**2 for x in x_arr)
    sum_y_sq = sum(y**2 for y in y_arr)

    denominator = ((n*sum_x_sq- sum_x**2) * (n*sum_y_sq-sum_y**2))**0.5
 
    # check for denominator 0 
    if denominator == 0:
        return 0  
    
    return numerator/denominator

def find_numerical_features(data, target: str, sorted_corr=False):
    # Check for target type 
    if not isinstance(target, str) :
        raise TypeError("Please pass in a string as your target.")
    
    num_cols = data.select_dtypes(include=[np.number]).columns
    corr_list = []
    
    for col in num_cols:
        target_corr = pearson_coeff_r(data[target], data[col])
        corr_list.append(target_corr)
        
    df = pd.DataFrame({"Column": num_cols, "Correlation With "+f"{target}": corr_list})
    
    if sorted_corr:
        sorted_df = df.sort_values(by="Correlation With "+f"{target}", ascending=False)
        return sorted_df, num_cols
    else:
        return df, num_cols


# link - https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
def find_categorical_features(data, target, sorted_corr=False):
    # Check for target type 
    if not isinstance(target, str) :
        raise TypeError("Please pass in a string as your target.")
    
    cat_cols = data.select_dtypes(exclude=[np.number]).columns
    corr_list = []
    
    
    for col in cat_cols:
        confusion_matrix = pd.crosstab(data[target],data[col])
        
        # Calculation based on Cramer's V
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
        rcorr = r-((r-1)**2)/(n-1)
        kcorr = k-((k-1)**2)/(n-1)
        corr_list.append(np.sqrt(phi2corr/min((kcorr-1),(rcorr-1))))

    df = pd.DataFrame({"Column": cat_cols, "Correlation With "+f"{target}": corr_list})
    
    if sorted_corr:
        sorted_df = df.sort_values(by="Correlation With "+f"{target}", ascending=False)
        return sorted_df, cat_cols
    else:
        return df, cat_cols


def iqr_range(arr):
    n = len(arr)
    # check for atleat 4 data points
    if n < 4:
        return None
    
    sorted_arr = sorted(arr)
    
    q1_index = int(n*0.25)
    q3_index = int(n*0.75)
    q1 = sorted_arr[q1_index]
    q3 = sorted_arr[q3_index]
    
    
    iq_range = q3 - q1
    
    iq_range_lower = q1 - 1.5 * iq_range
    iq_range_upper = q3 + 1.5 * iq_range
    

    
    return iq_range_lower, iq_range_upper


def calculate_accuracy(true_label, predicted_label):
    # check the length of labels
    if len(true_label) != len(predicted_label):
        raise Exception("Lengths do not match")
    
    correct_sum = 0
    for true, pred in zip(true_label, predicted_label):
        if true == pred:
            correct_sum += 1
            
    accuracy = correct_sum/len(true_label)
    
    return accuracy


def nn_classifier(X_train, y_train, num_nodes_l1, num_nodes_l2, lr, epochs, col_shape):
    
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_nodes_l1, activation='relu', input_shape=(col_shape,)),
        tf.keras.layers.Dense(num_nodes_l2, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    nn_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
        )
    history = nn_model.fit(X_train, y_train, epochs=epochs)

    return nn_model, history