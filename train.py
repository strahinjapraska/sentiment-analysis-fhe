import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 

def train():
    df = pd.read_parquet("data/dataset.parquet")
    X = np.vstack(df["embedding"].values)
    y = np.hstack(df["label"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  
    )

    model = LogisticRegression(
            solver = "lbfgs",
            max_iter=1000,
            multi_class="multinomial",
            random_state=42,
            class_weight="balanced",
    )
    
    model.fit(X_train, y_train)

    coefs = model.coef_
    intercepts =  model.intercept_

    print(model.score(X_test, y_test))

    print(str([[float(v) for v in w1] for w1 in coefs]).replace('[', '{').replace(']', '}') + ";")
    print(str([float(v) for v in intercepts]).replace('[', '{').replace(']', '}') + ";")




train() 