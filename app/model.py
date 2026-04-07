import joblib
import numpy as np
from pathlib import Path

SPECIES = ['setosa', 'versicolor', 'virginica']
FEATURES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
MODEL_VERSION = '1.0.0'


class MLModel:
    def __init__(self):
        self.model = None
        self.model_path = Path('models/model.pkl')

    def load(self):
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
        else:
            self._train_default_model()

    def _train_default_model(self):
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier

        iris = load_iris()
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(iris.data, iris.target)

        self.model_path.parent.mkdir(exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def predict(self, features):
        X = np.array(features).reshape(1, -1)
        pred = int(self.model.predict(X)[0])
        probs = self.model.predict_proba(X)[0]

        return {
            "prediction": pred,
            "species": SPECIES[pred],
            "confidence": float(probs[pred]),
            "probabilities": {
                SPECIES[i]: float(probs[i]) for i in range(len(SPECIES))
            }
        }

    def predict_batch(self, batch):
        return [self.predict(x) for x in batch]

    @property
    def is_loaded(self):
        return self.model is not None


# ✅ THIS LINE IS CRITICAL
ml_model = MLModel()