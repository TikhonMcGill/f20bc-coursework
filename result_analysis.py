import pandas as pd
import numpy as np

#load the cognitive and social results
results1 = pd.read_csv("cognitive&social_results.csv")
#load the global results
results2 = pd.read_csv("Global_results.csv")

combined_results = pd.concat([results1,results2],ignore_index=True)

#drop the previous botched indexing
combined_results.drop(columns=combined_results.columns[0],inplace=True)

standard_deviation = pd.DataFrame(columns=['profile_type','weight','average','standard_deviation'])

#we are going to split all the results into the profily_type and the weight, to get the standard deviation for the run
#for every type of profile
for t in combined_results['profile_type'].unique():
    #and every weight tested for that profile
    for weight in combined_results['weight'].unique():
        #make a temporary dataframe to filter the results dataframe
        #get the information for each run of the weight 
        temp_df = combined_results.loc[combined_results['profile_type'] == t]
        temp_df = temp_df.loc[temp_df['weight'] == weight]
        #get the standard deviation of the accuracy the runs 
        sd = temp_df['accuracy'].std()
        #create a standard deviation dataframe with all the information used for calculations removed and made into a simple easy to understand table with as the name implies, standard deviation
        standard_deviation.loc[len(standard_deviation)] = {'profile_type' : t, 'weight' : weight, 'average': temp_df['average'].mean(), 'standard_deviation' : sd}
        

print(standard_deviation)