from pydantic import BaseModel, ConfigDict, Field

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


class HeartFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    age: int = Field(ge=1, le=120, description="Возраст, лет")
    sex: int = Field(ge=0, le=1, description="Код пола: 0 или 1")
    cp: int = Field(ge=0, le=3, description="Код типа боли в груди")
    trestbps: float = Field(gt=0, le=300, description="Давление в покое, мм рт. ст.")
    chol: float = Field(gt=0, le=1000, description="Холестерин, мг/дл")
    fbs: int = Field(ge=0, le=1, description="Сахар натощак > 120 мг/дл: 0/1")
    restecg: int = Field(ge=0, le=2, description="Код ЭКГ в покое")
    thalach: float = Field(gt=0, le=250, description="Максимальная частота пульса")
    exang: int = Field(ge=0, le=1, description="Стенокардия при нагрузке: 0/1")
    oldpeak: float = Field(ge=0, le=10, description="Депрессия ST при нагрузке")
    slope: int = Field(ge=0, le=2, description="Код наклона ST")
    ca: int = Field(ge=0, le=4, description="Код количества крупных сосудов")
    thal: int = Field(ge=0, le=3, description="Код thal из исходного датасета")


class Prediction(BaseModel):
    target: int = Field(ge=0, le=1)
