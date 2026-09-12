"""Pydantic schemas for Activation Functions API requests and responses."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class ActivationFunctionsRequest(BaseModel):
    """Request schema for activation functions demonstration.

    Attributes:
        function_type: Primary activation function to demonstrate
        alpha: Leaky ReLU negative slope parameter
        input_range: Range for x-axis [min, max]
        compare_all: Whether to show all functions together
        num_points: Number of points to generate for visualization
    """

    function_type: str = Field(
        default='leaky_relu',
        description="Activation function to demonstrate"
    )
    alpha: float = Field(
        default=0.01,
        ge=0.0,
        le=0.3,
        description="Leaky ReLU negative slope (0 = ReLU, >0 = Leaky ReLU)"
    )
    input_range: List[float] = Field(
        default=[-10.0, 10.0],
        description="Range for x-axis [min, max]"
    )
    compare_all: bool = Field(
        default=True,
        description="Show all activation functions together"
    )
    num_points: int = Field(
        default=200,
        ge=50,
        le=1000,
        description="Number of points to generate for visualization"
    )

    @field_validator('function_type')
    @classmethod
    def validate_function_type(cls, v: str) -> str:
        """Validate activation function choice."""
        allowed = ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'elu', 'swish']
        if v not in allowed:
            raise ValueError(f"function_type must be one of {allowed}")
        return v

    @field_validator('input_range')
    @classmethod
    def validate_input_range(cls, v: List[float]) -> List[float]:
        """Validate input range."""
        if len(v) != 2:
            raise ValueError("input_range must contain exactly 2 values [min, max]")
        if v[0] >= v[1]:
            raise ValueError("input_range min must be less than max")
        if v[0] < -100 or v[1] > 100:
            raise ValueError("input_range must be within [-100, 100]")
        return v

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "function_type": "leaky_relu",
                "alpha": 0.01,
                "input_range": [-10.0, 10.0],
                "compare_all": True,
                "num_points": 200
            }
        }


class ActivationFunctionsResponse(BaseModel):
    """Response schema for activation functions demonstration.

    Attributes:
        success: Whether computation was successful
        function_data: Data for each activation function (x, y, derivative)
        comparison_table: Properties comparison table
        dead_neuron_demo: Dead neuron demonstration data
        visualization_data: Formatted data for frontend visualization
        execution_time_ms: Total execution time in milliseconds
        parameters_used: Parameters used for this computation
    """

    success: bool = Field(
        description="Whether computation completed successfully"
    )
    function_data: Dict[str, Dict[str, List[float]]] = Field(
        description="Data for each activation function with x, y, and derivative values"
    )
    comparison_table: List[Dict[str, Any]] = Field(
        description="Properties comparison table for all functions"
    )
    dead_neuron_demo: Dict[str, Any] = Field(
        description="Dead neuron demonstration comparing ReLU vs Leaky ReLU"
    )
    visualization_data: Dict[str, Any] = Field(
        description="Data formatted for frontend visualization"
    )
    execution_time_ms: float = Field(
        description="Total execution time in milliseconds"
    )
    parameters_used: Dict[str, Any] = Field(
        description="Parameters used for this computation"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "success": True,
                "function_data": {
                    "relu": {
                        "x": [-10.0, -5.0, 0.0, 5.0, 10.0],
                        "y": [0.0, 0.0, 0.0, 5.0, 10.0],
                        "derivative": [0.0, 0.0, 0.0, 1.0, 1.0]
                    }
                },
                "comparison_table": [
                    {
                        "function": "ReLU",
                        "range": "[0, ∞)",
                        "derivative_range": "{0, 1}",
                        "monotonic": True,
                        "zero_centered": False,
                        "gradient_when_negative": 0.0
                    }
                ],
                "dead_neuron_demo": {
                    "relu_gradient": 0.0,
                    "leaky_relu_gradient": 0.01,
                    "explanation": "ReLU has zero gradient for negative inputs (dead neurons)"
                },
                "visualization_data": {},
                "execution_time_ms": 12.34,
                "parameters_used": {
                    "function_type": "leaky_relu",
                    "alpha": 0.01
                }
            }
        }
