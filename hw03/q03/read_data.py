# %%
import os
import json
import pandas as pd

# %%
base_dir = "/home/amir/uni/nndl/hw03/q03/data/images"
def list_images(path):
    if os.path.isfile(path):
        if path.endswith(".jpg"):
            return [path]
        return []
    elif os.path.isdir(path):
        results = []
        for p in os.listdir(path):
            pp = os.path.join(path, p)
            results += list_images(pp)
        return results

images = list_images(base_dir)
# %%
df = pd.DataFrame(images, columns=["path"])
df["phase"] = df.path.str.split("/", expand=True)[9]
df["name"] = df.path.str.split("/", expand=True)[10]
df["image_id"] = df.name.str.rsplit(".", 1, expand=True)[0]
df.head(3)

# %%
annotations_file_location = "./data/annotations/instances_val.json"
with open(annotations_file_location, "r") as f:
    annotations = json.load(f)
annotations_df = pd.DataFrame(annotations["annotations"])

train_anot_location = "./data/images/.train.json"
with open(train_anot_location, "r") as file:
    train_anot = json.load(file)

# %%
train_images = []
for info in train_anot["information"]:
    train_images += info.split("\\")[-1]
print(len(train_images))
train_images = set(train_images)
print(len(train_images))

# %%
train_anot['information']
# %%

# df.image_id[0]
print(len(df))
len(df.image_id[df.phase=="val"].unique())

one = df.image_id[df.phase=="train"].unique()
one = set(one)
len(one)
# %%
annotations_df.image_id[0]
len(annotations_df.image_id.unique())
two = annotations_df.image_id.unique()
two = set(two)
len(two)

#%%
len(one.intersection(two))
#%%
df2 = pd.merge(annotations_df, df, how="right")

len(df2)
# 
# %%
