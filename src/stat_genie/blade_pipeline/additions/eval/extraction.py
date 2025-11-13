# imports
from stat_genie.blade_pipeline.llms.base import TextGenerator

def get_feature_transforms(llm_assistant: TextGenerator, transform_code: str,
                           feature_columns: list[str],
                           feature_description: str):
    """
    Given a list of feature columns, check if the columns are transformed in the
    transform code and return the code that performs the transformation.
    
    Args:
        llm_assistant (TextGenerator): The LLM assistant for the evaluation.
        transform_code (str): The code that performs the transformations.
        feature_columns (list[str]): The list of feature columns to check.
        
    Returns:
        dict: A dictionary of feature columns and the code that performs the transformation.
    """
    system_prompt = """You are an AI Data Analysis Assistant who is an expert at \
        performing data cleaning and preprocessing tasks."""
    transform_responses = []
    for feature_column in feature_columns:
        find_transform_prompt = f"""Given the following code:
            <Code>
            {transform_code}
            </Code>
            and the feature column:
            <Feature Column>
            {feature_column}
            </Feature Column>
            with description:
            <Feature Description>
            {feature_description}
            </Feature Description>
            determine if the column is transformed in the code. \
            If it is, return only the corresponding lines of code that perform the transformation. \
            If it is not, return "No transformation code found."
            """
        response = llm_assistant.generate([{"role": "system",
                                            "content": system_prompt},
                                           {"role": "user",
                                            "content": find_transform_prompt}])
        transform_responses.append(response)
    return transform_responses

def get_model_information(llm_assistant: TextGenerator, model_code: str):
    """
    Given modeling code, extract relevant information.
    
    Args:
        llm_assistant (TextGenerator): The LLM assistant for the evaluation.
        model_code (str): The code that defines the model.
        
    Returns:
        dict: A dictionary of model information, particularly model class.
    """
    
    system_prompt = """You are an AI Data Analysis Assistant who is an expert at \
        choosing, identifying, and implementing different types of ML models."""
    
    find_model_prompt = f"""Given the following code:
        <Code>
        {model_code}
        </Code>
        extract relevant information about the model. The returned value should be a dictionary with the following keys:
        1. "model_library": The library or framework used (e.g., "sklearn", "statsmodels", "pytorch", "tensorflow").
        2. "model_class": The specific model class or type (e.g., "LinearRegression", "RandomForestClassifier", "LogisticRegression").
        3. "model_parameters": Any parameters or hyperparameters that are set when instantiating or configuring the model.
        4. "model_formula_fitting_code": The code that defines the model formula and/or the code that fits/trains the model.
        
        The values of the dictionary should be strings.
        """
    
    response = llm_assistant.generate([{"role": "system",
                                        "content": system_prompt},
                                       {"role": "user",
                                        "content": find_model_prompt}])
    
    return response.text[0].content
