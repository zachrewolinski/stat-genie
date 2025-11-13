from stat_genie.blade_pipeline.llms.base import TextGenerator

def write_final_answer_code(llm_assistant: TextGenerator, task: list[str],
                            independent_variable: dict, dependent_variable: dict,
                            model_code: str, model_output):
    """
    In its attempt to answer the underlying question, GenAI analyst writes
    two functions: one which preprocesses the data and another that performs
    some sort of analysis.
    
    The modeling function written by the GenAI analyst does not explicitly
    answer the yes/no question posed in the task. The model code that it writes
    does something very close, which is fitting a model to the data that would
    answer the question with extra interpretations.
    
    The goal of this function is to complete analysis by interpreting the model
    output. We want to accomplish the task by asking an LLM to take in the model
    output and write code that extracts the final answer from the model output.
    This code must be written in python and have a consistent function header
    to make it easy to call later in the pipeline.
    
    Args:
        llm_assistant (TextGenerator): The LLM assistant for the evaluation.
        task (str): The task to be answered.
        independent_variable (dict): The independent variable to be used in the analysis.
        dependent_variable (dict): The dependent variable to be used in the analysis.
        model_code (str): The code that defines the model.
        model_output: The output of the model.
        
    Returns:
        str: A python function that reaches the conclusion of the task by
            extracting the final answer from the model output.
    """
    system_prompt = """You are an AI Data Analysis Assistant who is an expert at \
        drawing data-driven conclusions."""
    
    find_answer_prompt = f"""Given the following task/question:
        <Task>
        {task}
        </Task>
        
        The independent variable used in the analysis:
        <Independent Variable>
        {independent_variable}
        </Independent Variable>
        
        The dependent variable used in the analysis:
        <Dependent Variable>
        {dependent_variable}
        </Dependent Variable>
        
        The model code that was executed:
        <Model Code>
        {model_code}
        </Model Code>
        
        The output from running the modeling function (this is the raw model object, not an interpreted answer):
        <Model Output>
        {model_output}
        </Model Output>
        
        The modeling function does not explicitly answer the yes/no question posed in the task. 
        The model output is simply the raw model object returned by the model code (e.g., a fitted 
        statsmodels model, sklearn model, etc.). To answer the yes/no question, you need to:
        
        Write Python code that extracts relevant statistics from the model output object 
        (e.g., coefficients, p-values, confidence intervals, effect sizes) that relate 
        to the independent variable's effect on the dependent variable.
                
        The Python function needs to have the following function header:
        ```python
        def extract_final_answer(model_output):
            # Your code here to extract and interpret statistics from model_output
            # Return a dictionary with keys: "object", "description"
            pass
        ```
        
        The function should:
        - Take the model_output as input
        - Extract the necessary statistics from the model output object
        - Return a dictionary with:
            - "object": The actual value you would like to return (e.g. a coefficient, p-value, etc.)
            - "description": A brief explanation of the extracted statistics/return object and what it means in the context of the task
        
        Provide the complete function code that can be executed to extract the final answer.
        """
    
    response = llm_assistant.generate([{"role": "system",
                                        "content": system_prompt},
                                       {"role": "user",
                                        "content": find_answer_prompt}])
    
    return response.text[0].content
    
def make_conclusion(llm_assistant: TextGenerator, task: list[str],
                    independent_variable: dict, dependent_variable: dict,
                    model_code: str, interpretation_code: str,
                    interpretation_output: dict):
    """
    In its attempt to answer the underlying question, GenAI analyst writes
    three functions: one which preprocesses the data, another that performs
    some sort of analysis, and another that interprets the results.
    
    The goal of this function is to conclude the final answer by inspecting
    the interpretation of the model output. We want to accomplish the task by
    asking an LLM to take in the interpretation of the model output and
    explicitly answer the yes/no question posed in the task.
    
    Args:
        llm_assistant (TextGenerator): The LLM assistant for the evaluation.
        task (str): The task to be answered.
        independent_variable (dict): The independent variable to be used in the analysis.
        dependent_variable (dict): The dependent variable to be used in the analysis.
        model_code (str): The code that defines the model.
        interpretation_code (str): The code that interprets the model output.
        interpretation_output (dict): The output of the interpretation code.
        
    Returns:
        str: The final answer to the task. Must be either "Yes", "No", or "Not enough information".
    """
    system_prompt = """You are an AI Data Analysis Assistant who is an expert at \
        drawing data-driven conclusions from model summaries."""
    
    find_answer_prompt = f"""Given the following task/question:
        <Task>
        {task}
        </Task>
        
        The independent variable used in the analysis:
        <Independent Variable>
        {independent_variable}
        </Independent Variable>
        
        The dependent variable used in the analysis:
        <Dependent Variable>
        {dependent_variable}
        </Dependent Variable>
        
        The model code that was executed:
        <Model Code>
        {model_code}
        </Model Code>
        
        The model interpretation code that was executed:
        <Model Interpretation Code>
        {interpretation_code}
        </Model Interpretation Code>
        
        The model interpretation output:
        <Model Interpretation Output>
        {interpretation_output}
        </Model Interpretation Output>
        
        Analyze the model interpretation output and determine the final answer to the yes/no question posed in the task. 
        Return only a clear yes or no answer, with a brief justification if helpful. The final answer should be a dictionary with the following keys:
        1. "answer": The final answer to the question. Only valid options are "Yes", "No", or "Not enough information".
        2. "justification": A brief justification for the answer.
        """
    
    response = llm_assistant.generate([{"role": "system",
                                        "content": system_prompt},
                                       {"role": "user",
                                        "content": find_answer_prompt}])
    
    return response.text[0].content
