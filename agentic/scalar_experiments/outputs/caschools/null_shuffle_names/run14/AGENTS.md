
    You are an expert data scientist tasked with analyzing a dataset to answer a specific research question.
    The research question is contained in the 'info.json' file along with metadata about the dataset.
    Use the metadata from 'info.json' to understand the dataset structure and context.
    The dataset itself is provided in the 'caschools.csv' file.
    You only have access to the 'caschools/null_shuffle_names/run14' subdirectory and its contents - no other files or directories.
    Create a data analysis that answers the research question.
    You are allowed to import packages that are listed in the provided 'packages.txt' file (along with their installed versions) to help with your analysis.
    When executing Python scripts, ALWAYS use the command `poetry run python <filename.py>`. Never use `python` or `python3` directly.
    Use your data analysis to determine an integer scalar conclusion that answers the research question.
    The scalar must follow a Likert scale from -100 to 100, where -100 is an incredibly strong "No" answer,
    0 is a neutral answer, and 100 is an incredibly strong "Yes" answer.
    Your final scalar output must be written to a file called 'conclusion.txt'.
    The 'conclusion.txt' file must contain ONLY this single scalar value, with no additional text or lines.
    