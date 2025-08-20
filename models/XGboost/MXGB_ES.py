import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import os
class MultiTargetXGBRegressorWithEarlyStop:
    def __init__(self, 
                 n_estimators=1000, 
                 learning_rate=0.05,
                 early_stopping_rounds=20,
                 tree_method='hist',
                 predictor='cpu',
                 verbosity=0,
                 n_jobs=1):
        self.models = []
        self.best_iterations = []
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "early_stopping_rounds": early_stopping_rounds,
            "tree_method": tree_method,
            "predictor": predictor,
            "verbosity": verbosity,
            "n_jobs": n_jobs
        }
        self.save_dir = "result/figure/XGboost"
        os.makedirs(self.save_dir, exist_ok=True)

    def fit(self, X_train, Y_train, X_val, Y_val, verbose=False):
        print(XGBRegressor.__module__)
        n_outputs = Y_train.shape[1]
        self.models = []
        self.best_iterations = []
        
        for j in range(n_outputs):
            model = XGBRegressor(
                objective='reg:squarederror',
                n_estimators=self.params["n_estimators"],
                learning_rate=self.params["learning_rate"],
                early_stopping_rounds=self.params["early_stopping_rounds"],
                tree_method=self.params["tree_method"],
                predictor=self.params["predictor"],
                n_jobs=self.params["n_jobs"],
                verbosity=self.params["verbosity"],
                eval_metric="rmse"
            )
            
            model.fit(
            X_train, Y_train[:, j],
            eval_set=[(X_train, Y_train[:, j]), (X_val, Y_val[:, j])],

            
            verbose=verbose
            )
            # model.fit(
            #     X_train, Y_train[:, j],
            #     eval_set=[(X_val, Y_val[:, j])]

            # )

            self.models.append(model)
            self.best_iterations.append(model.best_iteration)

    def predict(self, X):
        if not self.models:
            raise ValueError("Model has not been trained yet.")
        return np.column_stack([model.predict(X) for model in self.models])

    def predict_argmin(self, X):
        Y_pred = self.predict(X)
        return np.argmin(Y_pred, axis=1)

    def get_best_iterations(self):
        return self.best_iterations
    
    def plot_learning_curves(self):
        save_dir = self.save_dir + "/learning_curves"
        os.makedirs(save_dir, exist_ok=True)  # 自动创建路径
        for idx, model in enumerate(self.models):
            evals_result = model.evals_result()
            if not evals_result:
                print(f"No evals_result for model {idx}")
                continue
            train_rmse = evals_result['validation_0']['rmse']
            val_rmse = evals_result['validation_1']['rmse']

            plt.figure(figsize=(8, 5))
            plt.plot(train_rmse, label='Train RMSE')
            plt.plot(val_rmse, label='Validation RMSE')
            plt.xlabel("Boosting Round")
            plt.ylabel("RMSE")
            plt.title(f"Learning Curve for Target {idx}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # 保存图片
            save_path = os.path.join(save_dir, f"target_{idx}.png")
            plt.savefig(save_path)
            print(f"Saved learning curve for target {idx} → {save_path}")

            plt.show()  # 如果你不想显示图像，可以注释掉这行

        return None