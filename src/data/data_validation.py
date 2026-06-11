# src/data_validation.py
import pandera as pa
import pandera.typing as pat


class IngestedMachineSchema(pa.DataFrameModel):
    # Standard column names match variable names directly
    UDI: pat.Series[int] = pa.Field(ge=1)
    Type: pat.Series[str] = pa.Field(isin=["L", "M", "H"])

    # Use 'alias' for columns with spaces, brackets, or special characters
    Product_ID: pat.Series[str] = pa.Field(
        alias="Product ID", str_matches=r"^[LMH]\d+$"
    )
    Air_temperature: pat.Series[float] = pa.Field(alias="Air temperature [K]", ge=0)
    Process_temperature: pat.Series[float] = pa.Field(
        alias="Process temperature [K]", ge=0
    )
    Rotational_speed: pat.Series[int] = pa.Field(alias="Rotational speed [rpm]", ge=0)
    Torque: pat.Series[float] = pa.Field(alias="Torque [Nm]", ge=0)
    Tool_wear: pat.Series[int] = pa.Field(alias="Tool wear [min]", ge=0)

    # Binary targets / failure flags
    Machine_failure: pat.Series[int] = pa.Field(alias="Machine failure", isin=[0, 1])
    TWF: pat.Series[int] = pa.Field(alias="TWF", isin=[0, 1])
    HDF: pat.Series[int] = pa.Field(alias="HDF", isin=[0, 1])
    PWF: pat.Series[int] = pa.Field(alias="PWF", isin=[0, 1])
    OSF: pat.Series[int] = pa.Field(alias="OSF", isin=[0, 1])
    RNF: pat.Series[int] = pa.Field(alias="RNF", isin=[0, 1])

    class Config:
        coerce = True  # Safely cast types (e.g., ints to floats if needed)
        strict = True  # Fail if upstream S3 data contains extra unmapped columns


@pa.check_types(lazy=True)
def train_model(data: pat.DataFrame[IngestedMachineSchema]):
    """
    This function will only ever receive a clean,
    validated DataFrame that fits the TrainingSchema.
    """
    print("Training initiated with validated data...")
    # Your training logic here...
    return "Model Trained"
