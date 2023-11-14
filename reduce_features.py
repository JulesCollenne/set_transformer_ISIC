import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


train_df = pd.read_csv("features/CNN_train.csv")
val_df = pd.read_csv("features/CNN_val.csv")
test_df = pd.read_csv("features/CNN_test.csv")
# Assuming df is your DataFrame with columns "feature0" to "feature512"
# You can replace df with your actual DataFrame variable

# Select features from "feature0" to "feature512"
oui = True
n_components = 100
tsne = PCA(n_components=n_components, random_state=42)
feature_columns = ["feature_" + str(i) for i in range(512)]

for df, name in zip((train_df, val_df, test_df), ("train", "val", "test")):
    X = df[feature_columns]

    # Initialize t-SNE with the desired number of components (200 in this case)
    # tsne = TSNE(n_components=200, random_state=42, )
    #
    # # Fit and transform the data
    # X_tsne = tsne.fit_transform(X)

    if oui:
        X_tsne = tsne.fit_transform(X)
        oui = False
    else:
        X_tsne = tsne.transform(X)

    # Create a new DataFrame with the reduced features
    columns_tsne = ["tsne_feature" + str(i) for i in range(n_components)]
    new_df = pd.DataFrame(X_tsne, columns=columns_tsne)

    # Optionally, you can concatenate other columns from the original DataFrame
    # For example, if you have a column named "target", you can concatenate it as follows:
    new_df = pd.concat([new_df, df["target"]], axis=1)
    new_df = pd.concat([new_df, df["image_name"]], axis=1)

    # Save the new DataFrame to a CSV file
    new_df.to_csv(f"features/new_CNN_{name}.csv", index=False)
