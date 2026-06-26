import pandera as pa
import pandera.typing as pat
from pydantic import BaseModel, Field


# 1. Pydantic Schema: Validates raw JSON structures coming into the REST API
class MachineInferencePayload(BaseModel):
    UDI: int
    product_id: str = Field(..., alias="Product ID")
    machine_type: str = Field(..., alias="Type")
    air_temp: float = Field(..., alias="Air temperature [K]")
    process_temp: float = Field(..., alias="Process temperature [K]")
    rotational_speed: int = Field(..., alias="Rotational speed [rpm]")
    torque: float = Field(..., alias="Torque [Nm]")
    tool_wear: int = Field(..., alias="Tool wear [min]")
    TWF: int = Field(..., alias="TWF")
    HDF: int = Field(..., alias="HDF")
    PWF: int = Field(..., alias="PWF")
    OSF: int = Field(..., alias="OSF")
    RNF: int = Field(..., alias="RNF")

    class Config:
        # Allows you to instantiate using either python attributes or the JSON aliases
        populate_by_name = True


# 2. Pandera Schema: Validates the final DataFrame in training pipelines and post-parsing
class MachineFeaturesSchema(pa.DataFrameModel):
    UDI: pat.Series[int] = pa.Field(ge=1)
    Product_ID: pat.Series[str] = pa.Field(
        alias="Product ID", str_matches=r"^[LMH]\d+$"
    )
    Type: pat.Series[str] = pa.Field(isin=["L", "M", "H"])
    Air_temperature: pat.Series[float] = pa.Field(
        alias="Air temperature [K]", ge=200, le=400
    )
    Process_temperature: pat.Series[float] = pa.Field(
        alias="Process temperature [K]", ge=200, le=400
    )
    Rotational_speed: pat.Series[int] = pa.Field(
        alias="Rotational speed [rpm]", ge=0, le=5000
    )
    Torque: pat.Series[float] = pa.Field(alias="Torque [Nm]", ge=0, le=200)
    Tool_wear: pat.Series[int] = pa.Field(alias="Tool wear [min]", ge=0, le=500)
    TWF: pat.Series[int] = pa.Field(alias="TWF", isin=[0, 1])
    HDF: pat.Series[int] = pa.Field(alias="HDF", isin=[0, 1])
    PWF: pat.Series[int] = pa.Field(alias="PWF", isin=[0, 1])
    OSF: pat.Series[int] = pa.Field(alias="OSF", isin=[0, 1])
    RNF: pat.Series[int] = pa.Field(alias="RNF", isin=[0, 1])

    class Config:
        coerce = True  # Auto-cast data types if possible
        strict = True  # Block unauthorized or mutated extra columns from passing
