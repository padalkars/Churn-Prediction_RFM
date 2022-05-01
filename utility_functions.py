#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[58]:


class GetStats():
    def __init__(self, data):
        self.data = data
        
    def missing_value_analysis(self):
        missing_value_count = self.data.isnull().sum().values
        missing_value_df = pd.DataFrame({"Features": self.data.columns.tolist(),
                                         "Missing Value Count": missing_value_count},
                                        columns = ["Features", "Missing Value Count"])
        
        missing_value_df["Missing Value Percentage"] = missing_value_df["Missing Value Count"]/self.data.shape[0]
        
        #Sort in descending order of percentages
        missing_value_df = missing_value_df.sort_values(by="Missing Value Percentage",                                                        ascending=False)
        return missing_value_df
    
    def get_descriptive_stats(self):
        data_types = self.get_data_types()
        
        #Continuous Variables(Mean, Median, Mode, Q25, Q50, Q75)
        continuous_vars = data_types.loc[(data_types['Data Types']=='float64') |                                            (data_types['Data Types']=='int64'), 'Features'].tolist()
        
        descriptive_stats = self.data[continuous_vars].describe().T
        descriptive_stats = descriptive_stats.reset_index()
        descriptive_stats = descriptive_stats.rename(columns={'index':'Features'})
        
        #Categorical Variables
        categorical_vars = data_types.loc[(data_types['Data Types']=='object'), 'Features'].tolist()
        category_count = list(map(lambda x: len(set(self.data[x])), categorical_vars))
        
        category_counts = pd.DataFrame({'Features': categorical_vars,                                        '# Categories': category_count},                                       columns = ['Features', '# Categories'])
        
        return descriptive_stats, category_counts
    
    def get_data_types(self):
        #A data frame with index as column names and correspondiong data types as colum 0  
        data_types = pd.DataFrame(self.data.dtypes) 
        data_types = data_types.reset_index() #A new column named 'index' is created
        data_types = data_types.rename(columns={'index':'Features', 0:'Data Types'})
        
        return data_types
    
    def outlier_analysis(self, cont_var):
        #Display the box_plot
        data_types = self.get_data_types()
        
        continuous_features = data_types.loc[(data_types['Data Types']=='float64') |                                            (data_types['Data Types']=='int64'), 'Features'].tolist()
        
        plt.boxplot(self.data[cont_var])
        
        plt.show()
    
    def categorical_distribution(self, cat_var):
        unique_cats = set(self.data[cat_var])
        cat_counts = {}
        for category in unique_cats:
            cat_counts[category] = self.data.loc[self.data[cat_var]==category, :].shape[0]
            
        distribution = pd.DataFrame({'Category':list(cat_counts.keys()),                                     'Counts': list(cat_counts.values())},                                     columns = ['Category', 'Counts'])
            
        #Percentage distribution
        total = sum(cat_counts.values())
        distribution['Percentage'] = distribution['Counts'] * 100/total #Broadcasting
        distribution['Percentage'] = distribution['Percentage'].apply(lambda x: round(x, 2))
        
        #Sort in descending order of counts
        distribution = distribution.sort_values(by='Counts', ascending=False)
    
        return distribution
    
    def driver(self):
        '''
        This function will call each of the above functions and 
        consolidate the statistics results.
        '''
        stats = self.missing_value_analysis()
        stats = pd.merge(stats, self.get_data_types(), on='Features', how='left')
        numeric_var_stats, categorical_stats = self.get_descriptive_stats()
        
        stats = pd.merge(stats, numeric_var_stats, on='Features', how='left')
        stats = pd.merge(stats, categorical_stats, on='Features', how='left')
        
        return stats

