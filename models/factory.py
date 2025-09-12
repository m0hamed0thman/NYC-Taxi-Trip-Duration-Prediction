from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

class ModelFactory:
    @staticmethod
    def create_model(name="ridge", degree=2):
        if name == "linear":
            return LinearRegression()
        elif name == "ridge":
            return Ridge(alpha=1)
        elif name == "lasso":
            return Lasso(alpha=0.01)
        elif name == "polynomial":
            # Polynomial Regression باستخدام LinearRegression
            return Pipeline([
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("linear", LinearRegression())
            ])
        elif name == "poly_ridge":
            # Polynomial + Ridge
            return Pipeline([
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("ridge", Ridge(alpha=1))
            ])
        elif name == "poly_lasso":
            # Polynomial + Lasso
            return Pipeline([
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("lasso", Lasso(alpha=0.01))
            ])
        else:
            raise ValueError(f"Unknown model type: {name}")
