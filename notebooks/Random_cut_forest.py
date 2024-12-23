#%%
import pandas as pd
#%%
sample = None
opf_data = pd.read_csv('../data/en.openfoodfacts.org.products.csv', sep='\t', encoding='utf-8', on_bad_lines='skip', nrows=sample)
opf_data.head()
#%%
opf_data.columns
# Write the columns to a file
opf_data.columns.to_list()

#%%
# Get the categories
# opf_data['categories'].unique()
opf_data['categories_tags'].unique().tolist()
# opf_data['categories_en'].unique().tolist()

#%%
# Filter the data where the nutriscore is valid
nutr_score_tgt = None

if nutr_score_tgt:
    opf_data_all = opf_data[opf_data['nutriscore_grade'].isin(['a','b','c','d','e'])]
    opf_data = opf_data_all[opf_data_all['nutriscore_grade'] == nutr_score_tgt]
else:
    opf_data = opf_data[opf_data['nutriscore_grade'].isin(['a','b','c','d','e'])]

opf_data.reset_index(drop=True, inplace=True)

opf_num_features = opf_data.filter(regex='_100g|score')
# replace Nan with zeros
opf_num_features.fillna(0, inplace=True)
opf_num_features.head()
#%%
important_nutrients = ['nutriscore_score', 'energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g', 'sodium_100g']
# important_nutrients = ['energy_100g', 'fat_100g', 'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g', 'sodium_100g']
opf_num_features = opf_num_features[important_nutrients]
# Convert nutriscore to float
opf_num_features['nutriscore_score'] = opf_num_features['nutriscore_score'].astype(float)
opf_num_features
#%%
# Get the nutriscore_score values to check if the conversion is successful
opf_num_features['nutriscore_score'].unique()
#%%
# One hot encode the features
data_target = opf_data.filter(regex='nutriscore_grade')
# data_target['nutriscore_grade'].replace({"a":4,"b":3,"c":2,"d":1,"e":0}, inplace=True)
data_target_one_hot = pd.get_dummies(data_target['nutriscore_grade'], prefix='nutriscore_grade')
# Replace True with 1 and False with 0
data_target_one_hot = data_target_one_hot.astype(int)
data_target_one_hot.head()
#%%
# Running Random Cut Forest 
import numpy as np
import rrcf

num_trees = 1000
n = opf_num_features.shape[0]
# tree_size = 256
tree_size = 64

# First idea is to concatenate the opf_num_features and data_target_one_hot along the columns
# Concatenate opf_num_features and data_target_one_hot along the columns
X = pd.concat([opf_num_features, data_target_one_hot], axis=1)
# The second idea is to use only the opf_num_features and filter on the nutriscore_grade
X = opf_num_features
# Convert to numpy array
X = X.to_numpy()


# Display the first few rows of the concatenated DataFrame
# X.head()

tree = rrcf.RCTree(X)

# Construct forest
forest = []
while len(forest) < num_trees:
    # Select random subsets of points uniformly from point set
    ixs = np.random.choice(n, size=(n // tree_size, tree_size),
                           replace=False)
    # Add sampled trees to forest
    trees = [rrcf.RCTree(X[ix], index_labels=ix) for ix in ixs]
    forest.extend(trees)

# Compute average CoDisp
avg_codisp = pd.Series(0.0, index=np.arange(n))
index = np.zeros(n)
for tree in forest:
    codisp = pd.Series({leaf : tree.codisp(leaf) for leaf in tree.leaves})
    avg_codisp[codisp.index] += codisp
    np.add.at(index, codisp.index.values, 1)
avg_codisp /= index
#%%
# Filter opf_data with the average CoDisp
opf_data['avg_codisp'] = avg_codisp
# filter columns on important_nutrients and other columns
opf_data = opf_data[['avg_codisp', 'nutriscore_score','nutriscore_grade', 'code','url','product_name','brands', 'categories_en'] + important_nutrients ]
opf_data.sort_values('avg_codisp', ascending=False, inplace=True)
opf_data
#%%
# Extract the outliers according to the average CoDisp and IQR
# Q1 = opf_data['avg_codisp'].quantile(0.25)
# Q3 = opf_data['avg_codisp'].quantile(0.75)
# IQR = Q3 - Q1
# outliers = opf_data[(opf_data['avg_codisp'] > (Q3 + 3 * IQR))]
outliers = opf_data[opf_data['avg_codisp'] == opf_data['avg_codisp'].max()]
outliers
#%%
# Get the feature importance
# Construct forest

d = X.shape[1]
forest = []
while len(forest) < num_trees:
    # Select random subsets of points uniformly from point set
    ixs = np.random.choice(n, size=(n // tree_size, tree_size),
                           replace=False)
    # Add sampled trees to forest
    trees = [rrcf.RCTree(X[ix], index_labels=ix) for ix in ixs]
    forest.extend(trees)


# Compute average CoDisp with the cut dimension for each point
dim_codisp = np.zeros([n,d],dtype=float)
index = np.zeros(n)
for tree in forest:
    for leaf in tree.leaves:
        codisp,cutdim = tree.codisp_with_cut_dimension(leaf)
        
        dim_codisp[leaf,cutdim] += codisp 

        index[leaf] += 1

avg_codisp = dim_codisp.sum(axis=1)/index

#codisp anomaly threshold and calculate the mean over each feature
feature_importance_anomaly = np.mean(dim_codisp[avg_codisp>50,:],axis=0)
#create a dataframe with the feature importance
df_feature_importance = pd.DataFrame(feature_importance_anomaly,columns=['feature_importance'])
df_feature_importance
#%%
# Map the feature importance to the important_nutrients
df_feature_importance.index = important_nutrients
df_feature_importance.sort_values('feature_importance', ascending=False, inplace=True)
df_feature_importance
#%%
import rrcf
tree = rrcf.RCTree()
print(dir(tree))
#%%
