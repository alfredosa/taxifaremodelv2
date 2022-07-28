from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from . import get_data,utils,encoders

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', encoders.DistanceTransformer()),
            ('stdscaler', StandardScaler())
            ])
        time_pipe = Pipeline([
                    ('time_enc', encoders.TimeFeaturesEncoder('pickup_datetime')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore'))
                ])
        preproc_pipe = ColumnTransformer([
                ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
                ('time', time_pipe, ['pickup_datetime'])
            ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        self.pipeline = pipe

    def run(self):
        """set and train the pipeline"""
        return self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return utils.compute_rmse(y_pred, y_test)


if __name__ == "__main__":
    df = get_data.clean_data(get_data.get_data(nrows=10,local=True))
    y = df.pop("fare_amount")
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    trainer = Trainer(X_train, y_train)
    trainer.set_pipeline()
    trainer.run()
    eval = trainer.evaluate(X_test,y_test)
    print(eval)
