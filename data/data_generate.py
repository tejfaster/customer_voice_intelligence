import pandas as pd

# loading files
df1 = pd.read_csv("archive/1429_1.csv",low_memory=False)
df2 = pd.read_csv("archive/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv", low_memory=False)
df3 = pd.read_csv("archive/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv", low_memory=False)

# Combine
combined_df = pd.concat([df1,df2,df3],ignore_index=True)

# save
combined_df.to_csv("amazon_reviews.csv",index=False)

print(combined_df.shape)