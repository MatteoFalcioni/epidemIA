from __future__ import annotations
from pyexpat import model
from typing import TypedDict
from langchain_core.tools import tool
from langgraph.types import Command
from typing_extensions import Annotated, Literal, NotRequired
from datetime import datetime, timezone
from langchain_core.messages import ToolMessage
from langchain.tools import ToolRuntime
import json


import pandas as pd
from . import models
from . import metrics
import importlib
import pkgutil


REQUIRED_ATTRS = {
    "name",
    "PARAMETER_FIELDS",
    "EXPLAIN_PARAMETERS",
    "fit_parameters_defaults",
    "fixed_parameters_defaults",
    "bounds",
    "incidence",
    "fit_model"
}



def _load_model_module(model: str):
    """Import models.<MODEL> and validate it exposes the full contract."""

    # trova il nome reale del modulo confrontando in lowercase
    real_name = next(
        (m.name for m in pkgutil.iter_modules(models.__path__)
         if m.name.lower() == model.lower()),
        None
    )

    if real_name is None:
        raise ImportError(f"No module found for model '{model}'")

    mod = importlib.import_module(f".models.{real_name}", package=__package__)

    missing = REQUIRED_ATTRS - set(dir(mod))
    if missing:
        raise ImportError(f"Model '{model}' is missing: {missing}")

    return mod


def _merge_with_defaults(defaults: dict, user_params: dict | None) -> dict:
    if user_params is None:
        return defaults.copy()
    invalid = set(user_params) - set(defaults)
    if invalid:
        raise ValueError(f"Unknown parameters: {invalid}. Valid: {set(defaults)}")
    return {**defaults, **user_params}



def _build_models_schema(available_models: list[str]) -> tuple[dict, dict]:
    """
    Load every model module and collect their PARAMETER_FIELDS + defaults.
    Returns two nested dicts the LLM can read from the docstring:
    - fit_schema: parameters tunable at fit time
    - fixed_schema: fixed parameters set at model construction
    """
    fit_schema = {}
    fixed_schema = {}

    for model_name in available_models:
        mod = _load_model_module(model_name)

        fit_schema[model_name] = {
            field: {
                "type": typ.__name__,
                "default": mod.fit_parameters_defaults[field],
                "meaning": mod.EXPLAIN_PARAMETERS[field],
            }
            for field, typ in mod.PARAMETER_FIELDS.items()
            if field in mod.fit_parameters_defaults
        }

        fixed_schema[model_name] = {
            field: {
                "type": typ.__name__,
                "default": mod.fixed_parameters_defaults[field],
                "meaning": mod.EXPLAIN_PARAMETERS[field],
            }
            for field, typ in mod.PARAMETER_FIELDS.items()
            if field in mod.fixed_parameters_defaults
        }

    return fit_schema, fixed_schema


def make_fit_tools(models_list: list[str]):
    """
    Build and return a LangGraph @tool with:
      - model: Literal[...] auto-built from available_models
      - model_parameters: dict | None  (schema injected into docstring)

    Call once at startup:
        fit_model_from_csv = make_fit_tool(["SIR", "SEIR", "GAMMASIR"])
    """
    
    fit_schema, fixed_schema = _build_models_schema(models_list)
    fit_schema_json = json.dumps(fit_schema, indent=2)
    fixed_schema_json = json.dumps(fixed_schema, indent=2)


    @tool
    def fit_and_forecast(runtime: ToolRuntime,
                        start_date : str ,
                        csv_path: Annotated[str, "Path to the CSV file containing the full data range of incidence data before start date."],
                        model_name : Annotated[str, "Name of the epidemiological model to fit."] = "SIR",
                        col_name : Annotated[str, "Name of the column in the CSV file containing incidence data."] = "incidence",
                        model_initial_parameters: dict | None = None,
                        model_fixed_parameters: dict | None = None,
                        metric: Literal["rmse"] = "rmse",
                        prediction_days : Annotated[int, "Number of days to predict."] = 0,
                        rolling_window : int = 7 
                        ) -> Command:
    
        """Predict incidence values using the chosen epidemiological model.

        Before calling, use get_model_info(model) to discover the right model parameters. You only need to pass the ones you want to override,
        the rest use their defaults.

        It returns dates even before start_date, which are the one used to fit the model.

        Args:
        runtime: Tool runtime context (used for message routing and call tracking).
        start_date: Start date for the forecast in YYYY-MM-DD format.
        csv_path: Path to the CSV file containing the full data range of incidence data before start date, with datetime index and "incidence" column.
        model_name: Model identifier string. Determines which
            model spec and parameter schema will be used. Default is 'SIR'
        model_initial_parameters: Optional per-model initial guesses. Missing fields are
            filled from model defaults. This argument is inferred from get_model_info output, so you can just pass the fields you want to override.
        model_fixed_parameters: Optional per-model fixed parameters. These are not fitted but are used in the simulation.
        metric: Metric name for optimization. NOTE: Currently only "rmse" is supported.
        prediction_days: Number of days to predict forward.
        col_name: Name of the column in the CSV file containing incidence data (default: "incidence").
        rolling_window: Window size for rolling average smoothing (default: 7 days).

    Returns:
        Command: Result command with two updates:
            - "messages": ToolMessage with JSON-serialized result_dict.
            - "simulations": list containing result_dict for state propagation.
    """
        

        try:
            #model = models.available_models[model_name]
            model = _load_model_module(model_name)
        except KeyError:
            return {'Message' : f'Error, no model named {model_name} available in the models/ folder.'}

        try:
            metric = metrics.available_metrics[metric].metric
        except KeyError:
            return {'Message' : f'Error, no metric named {metric} available in the metrics/ folder.'}

        # Load dataset.
        
        df = pd.read_csv(csv_path, index_col = 0, parse_dates = True)
        
        # Cast start_date to datetime. 
        start_date = pd.to_datetime(start_date)

        # Cast simulation simulation to Timedelta and get end_date. 
        sim_timedelta = pd.Timedelta(days=prediction_days)
        end_date = start_date + sim_timedelta 
        
        # Check that start and end of fit/simulation are within bounds.
        if (start_date <= df.index.min()) or (start_date  >= df.index.max()):
            return {'Message' : f'Error, start date out of data range.'}
            
        # Compute overall mean and standard deviation to compute threshold.
        global_mean = df[col_name].to_numpy().mean()
        global_std  = df[col_name].to_numpy().std()
        #global_std = 0

        threshold = global_mean + global_std
        
        df['smooth'] = df[col_name].rolling(rolling_window, 
                                    center = False, 
                                    min_periods = 1).mean()
        
        # Check if the user asked for prediction during an outbreak.
        if df.loc[start_date].smooth < threshold:
            return {'Message' : f'No epidemic increase in the incidence detected at required time. You should report to the supervisor that the prediction cannot be performed because there is no epidemic increase in the incidence detected at required time.'}


        df = df[df.index <= start_date]
        # If we are here it means an epidemic is taking place.
        # We use as the start of the fitting procedure a point 2 weeks before
        # the crossing of the alert level.
        closest_xing = df[(df.smooth >= threshold) 
                        & (df.smooth.shift(1) < threshold)].index[-1]
        fit_begin = closest_xing - pd.Timedelta(days = 14) 

        # Extract the data to fit on.
        
        df_fit = df[(df.index >= fit_begin) & (df.index <= start_date)]

        fit_incidence = df_fit.smooth.to_numpy()



        fit = model.fit_model(fit_incidence, metric,
                            model_initial_parameters, model_fixed_parameters)
        

        
        sim_incidence, dt_index = model.incidence(fit_begin, end_date, 
                                        fit['fitted_parameters'])
        
        fit_parameters = {k: (float(v) if hasattr(v, 'item') else v) for k, v in fit['fitted_parameters'].items()}
        predicted_incidence_list = [float(value) for value in sim_incidence]
        incidence_dates = [str(date.date()) for date in dt_index]

        result_dict = {
#            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "csv_path": csv_path,
            "parameters": fit_parameters,
            'R0': float(fit['R0']),
            "rmse": float(fit['metric_minimum']),
            "success": bool(fit['success']),
            "message": str(fit['status']),
            "predicted_incidence": predicted_incidence_list,
            "predicted incidence dates" : incidence_dates,
            "n_days_predicted": prediction_days,
        }


        return Command(
            update={
                "messages": [ToolMessage(content=json.dumps(result_dict), tool_call_id=runtime.tool_call_id)],
                "simulations": [result_dict],
            }
        )
        

    docstring_suffix = f"""
    \nmodel_name must be one of: {models_list}

    model_initial_parameters per model:
    {fit_schema_json}
    \nmodel_fixed_parameters per model:
    {fixed_schema_json}
    """
    
    fit_and_forecast.__doc__ = (fit_and_forecast.__doc__ + docstring_suffix)


    @tool
    def incidence(runtime: ToolRuntime,
                  csv_path: Annotated[str, "Path to the CSV file containing the full data range of incidence data before start date."],
                start_date : Annotated[str, "Start date for the forecast in YYYY-MM-DD format."],
                  end_date : str,
                  col_name : Annotated[str, "Name of the column in the CSV file containing incidence data."] = "incidence",
                model_name : Annotated[str, "Name of the epidemiological model to use."] = "SIR",
                rolling_window : int = 7 ,
                 parameters : dict | None = None):
        """Simulate incidence values using the chosen epidemiological model and given parameters.
        
                Before calling, use get_model_info(model) to discover the right model parameters. You only need to pass the ones you want to override,
                the rest use their defaults.
        
                Args:
                runtime: Tool runtime context (used for message routing and call tracking).
                csv_path: Path to the CSV file containing the full data range of incidence data before start date, with datetime index and "incidence" column.
                start_date: Start date for the forecast in YYYY-MM-DD format.
                end_date: End date for the forecast in YYYY-MM-DD format.
                col_name: Name of the column in the CSV file containing incidence data (default: "incidence").
                model_name: Model identifier string. Determines which
                    model spec and parameter schema will be used. Default is 'SIR'
                rolling_window: Window size for rolling average smoothing (default: 7 days).
                parameters: Optional per-model parameters. Missing fields are
                    filled from model defaults. This argument is inferred from get_model_info output, so you can just pass the fields you want to override.
        
            Returns:
                Command: Result command with two updates:
                    - "messages": ToolMessage with JSON-serialized result_dict.
                    - "simulations": list containing result_dict for state propagation.
            """

        # Load dataset.
                
        df = pd.read_csv(csv_path, index_col = 0, parse_dates = True)
        
        # Cast start_date to datetime. 
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Check that start and end of fit/simulation are within bounds.
        if (start_date <= df.index.min()) or (start_date  >= df.index.max()):
            return {'Message' : f'Error, start date out of data range.'}
            
        # Compute overall mean and standard deviation to compute threshold.
        global_mean = df[col_name].to_numpy().mean()
        global_std  = df[col_name].to_numpy().std()
        #global_std = 0

        threshold = global_mean + global_std
        
        df['smooth'] = df[col_name].rolling(rolling_window, 
                                    center = False, 
                                    min_periods = 1).mean()
        
        # Check if the user asked for prediction during an outbreak.
        if df.loc[start_date].smooth < threshold:
            return {'Message' : f'No epidemic increase in the incidence detected at required time. You should report to the supervisor that the prediction cannot be performed because there is no epidemic increase in the incidence detected at required time.'}


        df = df[df.index <= start_date]
        # If we are here it means an epidemic is taking place.
        # We use as the start of the fitting procedure a point 2 weeks before
        # the crossing of the alert level.
        closest_xing = df[(df.smooth >= threshold) 
                        & (df.smooth.shift(1) < threshold)].index[-1]
        fit_begin = closest_xing - pd.Timedelta(days = 14) 
        
        try:
            model = _load_model_module(model_name)
        except KeyError:
            return {'Message' : f'Error, no model named {model_name} available in the models/ folder.'}



        parameters = _merge_with_defaults(model.fit_parameters_defaults, parameters)

        for par in parameters.values():
            if par is None:
                return Command(
                            update={
                                "messages": [ToolMessage(content='Error: A required parameter is equal to None. Please ' \
                                'provide a value for every parameter.', tool_call_id=runtime.tool_call_id)]
                            }
                        )

        
        sim_incidence, dt_index = model.incidence(fit_begin, end_date, parameters)

        predicted_incidence_list = [float(value) for value in sim_incidence]
        incidence_dates = [str(date.date()) for date in dt_index]

        result_dict = {
        #            "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": model_name,
                    "predicted_incidence": predicted_incidence_list,
                    "predicted incidence dates" : incidence_dates
                }
        
        
        return Command(
                    update={
                        "messages": [ToolMessage(content=json.dumps(result_dict), tool_call_id=runtime.tool_call_id)],
                        "simulations": [result_dict],
                    }
                )


    docstring_suffix = f"""\nmodel_name must be one of: {models_list}
    
        parameters per model:
        {fit_schema_json}"""

    incidence.__doc__ = (incidence.__doc__ + docstring_suffix)
    

    return fit_and_forecast, incidence



def make_model_info_tool(available_models: list[str]):
    """
    Build a get_model_info tool the LLM calls BEFORE fit_model_from_csv
    to discover which parameters a model accepts.
    """
    

    ModelLiteral = Literal[tuple(available_models)]  # type: ignore[valid-type]
    
    @tool
    def get_model_info(model: Annotated[str, f'Model name'] = "SIR") -> dict:
        """
        Return the available parameters and their defaults for a given model.
        Always call this before fit_model_from_csv to know what you can set.

        model: Model name. Default is 'SIR'
        """
        mod = _load_model_module(model)
        return {
            "model": model,
            "parameters": {
                field: {
                "type": typ.__name__,
                "default": mod.fit_parameters_defaults[field] if field in mod.fit_parameters_defaults else mod.fixed_parameters_defaults[field],
                "meaning": mod.EXPLAIN_PARAMETERS[field]
            }
                for field, typ in mod.PARAMETER_FIELDS.items()
            },
        }

    docstring_suffix = f"""
    model must be one of: {available_models}
    """
    get_model_info.__doc__ = (
    get_model_info.__doc__ + docstring_suffix)

    return get_model_info





def discover_models() -> list[str]:
    """Find all modules in the models/ directory and return their names."""
    return [mod.name
        for mod in pkgutil.iter_modules(models.__path__)
        if not mod.name.startswith("_")  # esclude __init__, _utils, ecc.
    ]




def swab_result_mapper(result_string, col_value):
    """
    Map a swab result string to a simplified code.

    Args:
        result_string (str): Swab result string.

    Returns:
        str: Simplified code ('p' for positive, 'n' for negative, 'i' for inconclusive).
    """
    return 'p' if col_value in result_string else 'n'
    

@tool
def csv_writer(runtime: ToolRuntime,
               csv_path : Annotated[str, "Path to the CSV file to write the results to."],
               col_name : Annotated[str, "Name of the column that has to be counted"] = 'ESITO TAMPONE',
               col_value : Annotated[str, "Value of the column to count"] = 'Positivo') -> Command:
    
    """Create a new csv file with the daily incidence of a given value in a given column, taking data from the csv_path file.
    
    Args:
    csv_path: Path to the CSV file containing the full data range of incidence data before start date, with datetime index and a column with name col_name.
    col_name: Name of the column in the CSV file to count values from.
    col_value: Value of the column to count for incidence."""
    
    df = pd.read_csv(csv_path, index_col = 0, parse_dates = True)

    unstacked = df.groupby([pd.Grouper(freq = 'D'), col_name])['Totali Accessi'].sum().unstack(0)
    unstacked.index = unstacked.index.map(lambda s: 'p' if col_value in s else 'n')
    daily_positive_swabs = unstacked.loc['p'].sum()

    try:
        daily_positive_swabs = pd.DataFrame(daily_positive_swabs, columns=['incidence'])

    except ValueError as e:
        daily_positive_swabs = unstacked.loc['p']
        daily_positive_swabs = pd.DataFrame(daily_positive_swabs)
        

    daily_positive_swabs.to_csv('context/daily_incidence.csv')

    return Command(
            update={
                "messages": [ToolMessage(content=json.dumps('Daily incidence data saved in context/daily_incidence.csv'), tool_call_id=runtime.tool_call_id)],
            }
        )
