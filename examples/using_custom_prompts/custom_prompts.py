# FIRST PROMPT
SYSTEM_PROMPT_1 = """You are an AI Data Analysis Assistant who is an expert at \
writing an end-to-end scientific analysis given a research question and a dataset. \
You are skilled at understanding a research question, relecting on the data and relevant domain \
knowledge, and representing this conceptual knowledge in a statistical model. \
Key to this modeling process is formalizing the conceptual model, which includes \
variables and their relationships that are relevant to the domain and data."""


INSTRUCTION_PROMPT_1 = """<Instruction> 
Given the research question, dataset \
formulate the conceptual model and write an analysis including all necessary \
data transformations and a statistical model to answer the research question. 
</Instruction>

<Format Instructions>
You will return 3 things:
1. The conceptual variables which includes a natural language description of the variables, the variable \
type (i.e., Independent, Dependent, Control), and any relationships between the variables. Each variable should also \
describe which column(s) in the final dataframe (output of the transform function and used in the statistical model) it is associated with. \
IMPORTANT: The column names in the conceptual variables should be the EXACT column names used in the model code. \
    
2. The transform function which follows the which will take the original dataframe \
and return the dataframe after all transformations. \
The returned dataframe should include all the columns that are necessary for \
the subsequent statistical modeling. \
If you are changing any values of columns or deriving new columns, \
you should add this as a new column to the dataframe. \
    
3. The model function which will take the transformed dataframe \
and run a statistical model on it. The model function should return the results of the model.

The following libraries are already imported but you can import any popular libraries you need:
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt

Here is the code template for the transform function:
```python
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Your code here
    return df
```
Here is the code template for the model function:
```python
def model(df: pd.DataFrame) -> Any:
    # Your code here
    return results
```

Please return the conceptual variables, the transform function, and the model function in the format specified below:
{format_instructions}
</Format Instructions>
"""

EXAMPLE_1 = """<Example>
Research Question: {research_question_ex}
Dataset Schema: {dataset_schema_ex}
Result: {result_ex}
</Example>
"""

POST_FIX_1 = """Research Question: {research_question}
Dataset Schema: {dataset_schema}
Result: """

# SECOND PROMPT
SYSTEM_PROMPT_2 = """You are an AI Data Analysis Assistant who is an expert at \
writing an end-to-end scientific analysis given a research question and a dataset. \
You are skilled at understanding a research question, relecting on the data and relevant domain \
knowledge, and representing this conceptual knowledge in a statistical model. \
Key to this modeling process is formalizing the conceptual model, which includes \
variables and their relationships that are relevant to the domain and data."""


INSTRUCTION_PROMPT_2 = """<Instruction> 
Given the research question, dataset \
formulate the conceptual model and write an analysis including all necessary \
data transformations and a statistical model to answer the research question. 
</Instruction>

<Format Instructions>
You will return 2 things:

1. The transform function which follows the which will take the original dataframe \
and return the dataframe after all transformations. \
The returned dataframe should include all the columns that are necessary for \
the subsequent statistical modeling. \
If you are changing any values of columns or deriving new columns, \
you should add this as a new column to the dataframe. \
    
2. The model function which will take the transformed dataframe \
and run a statistical model on it. The model function should return the results of the model. \
The results of the model should be a single number (e.g. model coefficient, R-squared value, p-value, etc.) \
that summarizes the key finding/conclusion from the model. \

The following libraries are already imported but you can import any popular libraries you need:
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt

Here is the code template for the transform function:
```python
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Your code here
    return df
```
Here is the code template for the model function:
```python
def model(df: pd.DataFrame) -> Any:
    # Your code here
    return result
```

Please return the conceptual variables, the transform function, and the model function in the format specified below:
{format_instructions}
</Format Instructions>
"""

EXAMPLE_2 = """<Example>
Research Question: {research_question_ex}
Dataset Schema: {dataset_schema_ex}
Result: {result_ex}
</Example>
"""

POST_FIX_2 = """Research Question: {research_question}
Dataset Schema: {dataset_schema}
Result: """