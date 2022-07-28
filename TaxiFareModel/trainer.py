from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
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
        self.mlflowuri = "https://mlflow.lewagon.ai/"
        self.experiment_name = "DEU_BER_alfredosa_model_TaxiFare"
        # 🚨 replace with your country code, city,
        # github_nickname and model name and version

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
        evalres = utils.compute_rmse(y_pred, y_test)
        self.mlflow_run()
        self.mlflow_log_metric("rmse", evalres)
        self.mlflow_log_param("model", "LinearRegression")
        return

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.mlflowuri)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

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
