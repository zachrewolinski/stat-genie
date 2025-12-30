# imports
from stat_genie.blade_pipeline.llms.config import llm
from joblib import Parallel, delayed
from copy import deepcopy

def get_feature_transforms(llm_provider: str, llm_model: str,
                           transform_code: str,
                           feature_columns: list[str],
                           feature_description: str):
    """
    Given a list of feature columns, check if the columns are transformed in the
    transform code and return the code that performs the transformation.
    
    Args:
        llm_provider (str): The provider of the LLM.
        llm_model (str): The model of the LLM.
        transform_code (str): The code that performs the transformations.
        feature_columns (list[str]): The list of feature columns to check.
        
    Returns:
        dict: A dictionary of feature columns and the code that performs the
              transformation.
    """
    system_prompt = """You are an AI Data Analysis Assistant who is an expert \
        at performing data cleaning and preprocessing tasks."""
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
            determine if the column is transformed in the code. If it is, return
            only the corresponding lines of code that perform the
            transformation. If it is not, return "No transformation code found."
            """
        llm_assistant = llm(provider=llm_provider, model=llm_model)
        response = llm_assistant.generate([{"role": "system",
                                            "content": system_prompt},
                                           {"role": "user",
                                            "content": find_transform_prompt}])
        transform_responses.append(response)
    return transform_responses

def get_model_information(llm_provider: str, llm_model: str, model_code: str):
    """
    Given modeling code, extract relevant information.
    
    Args:
        llm_provider (str): The provider of the LLM.
        llm_model (str): The model of the LLM.
        model_code (str): The code that defines the model.
        
    Returns:
        dict: A dictionary of model information, particularly model class.
    """
    
    system_prompt = """You are an AI Data Analysis Assistant who is an expert \
        at choosing, identifying, and implementing different types of ML models.
        """
    
    find_model_prompt = f"""Given the following code:
        <Code>
        {model_code}
        </Code>
        extract relevant information about the model. The returned value should
        be a dictionary with the following keys:
        1. "model_library": The library or framework used (e.g., "sklearn",
                            "statsmodels", "pytorch", "tensorflow").
        2. "model_class": The specific model class or type (e.g.,
                          "LinearRegression", "RandomForestClassifier",
                          "LogisticRegression").
        3. "model_parameters": Any parameters or hyperparameters that are set
                               when instantiating or configuring the model.
        4. "model_formula_fitting_code": The code that defines the model formula
                                         and/or the code that fits/trains the
                                         model.
        
        The values of the dictionary should be strings.
        """
        
    llm_assistant = llm(provider=llm_provider, model=llm_model)
    
    response = llm_assistant.generate([{"role": "system",
                                        "content": system_prompt},
                                       {"role": "user",
                                        "content": find_model_prompt}])
    
    return response.text[0].content

def format_features(multirun_analyses: dict, num_runs: int,
                    llm_provider: str, llm_model: str, n_jobs: int=-1):
    
    # helper function to process a single run
    def process_single_run(i):
        
        # create internal dict for analysis features
        run_features = {}
        
        # get the features from each analysis
        # this should include the independent and control variables
        ind_vars = deepcopy(multirun_analyses['analyses'][str(i)]['cvars']['ivs'])
        contr_vars = deepcopy(multirun_analyses['analyses'][str(i)]['cvars']['controls'])
        response_vars = deepcopy(multirun_analyses['analyses'][str(i)]['cvars']['dv'])

        # get any lines from the transform code that represent transformations
        # of the independent or control variables
        transform_code = multirun_analyses['analyses'][str(i)]['transform_code']
        
        # for each variable in ind_vars, check if it is transformed
        # in the transform_code by using an LLM assistant
        for dict_idx, var in enumerate(ind_vars):
            transform_responses = get_feature_transforms(llm_provider,
                                                         llm_model,
                                                         transform_code,
                                                         var['columns'],
                                                         var['description'])
            ind_vars[dict_idx]['transform_code'] = \
                [response.text[0].content for response in transform_responses]
            
        # save updated independent variables in features dict
        run_features['independent_variables'] = ind_vars
        
        # take same approach for control variables
        for dict_idx, var in enumerate(contr_vars):
            transform_responses = get_feature_transforms(llm_provider,
                                                         llm_model,
                                                         transform_code,
                                                         var['columns'],
                                                         var['description'])
            contr_vars[dict_idx]['transform_code'] = \
                [response.text[0].content for response in transform_responses]
        
        # save updated control variables in features dict
        run_features['control_variables'] = contr_vars

        # for each variable in response_vars, check if it is transformed
        # in the transform_code by using an LLM assistant
        transform_responses = get_feature_transforms(llm_provider,
                                                     llm_model,
                                                transform_code,
                                                response_vars['columns'],
                                                response_vars['description'])
        response_vars['transform_code'] = [response.text[0].content \
            for response in transform_responses]

        # save updated response variables in features dict
        run_features['response_variables'] = response_vars
        
        return i, run_features
    
    # parallelized looping through analyses
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_run)(i) for i in range(num_runs)
    )

    # convert results list to dictionary
    features = {i: run_features for i, run_features in results}
    
    # return the features dict
    return features

def format_model_info(multirun_analyses: dict, num_runs: int,
                      llm_provider: str, llm_model: str, n_jobs: int=-1):
    
    # create dict to store model information
    model_info = {}
    
    # helper function to process a single run
    def process_single_run(i):
        
        # get the model code
        model_code = multirun_analyses['analyses'][str(i)]['m_code']
        
        # get the model information using an LLM assistant
        model_information = get_model_information(llm_provider,
                                                  llm_model,
                                                  model_code)
        
        return i, model_information
    
    # parallelized looping through analyses
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_run)(i) for i in range(num_runs)
    )
    model_info = {i: info for i, info in results}
        
    # return the model information dict
    return model_info
    